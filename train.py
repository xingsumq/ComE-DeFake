import os
import time
import copy
import torch.optim as optim
from utility.utils import *
from utility.metric import *
from model.ComEDeFake import *
from utility.globals import *
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import random


cfg = get_config()
if torch.cuda.is_available():
    print("cuda is available")
else:
    print("running with cpu")

fts_node, fts_edge, H, lbls, idx_train, idx_val, idx_test = load_data()
fts_node = fts_node.astype(float)

fts_edge = fts_edge.astype(float)
# fts_edge = torch.eye(H.shape[1])

GV = generate_G_from_H(H)
GE = generate_G_from_H(H.transpose())
n_class = int(lbls.max()) + 1
device = torch.device('cuda:0' if int(args_global.gpus) >= 0 else 'cpu')
num_news = fts_edge.shape[0]

# transform data to device
fts_node = torch.Tensor(fts_node).to(device)
fts_edge = torch.Tensor(fts_edge).to(device)
H = torch.Tensor(H).to(device)
lbls = torch.Tensor(lbls).squeeze().long().to(device)
GV = torch.Tensor(GV).to(device)
GE = torch.Tensor(GE).to(device)
idx_train = torch.Tensor(idx_train).long()
idx_val = torch.Tensor(idx_val).long()
idx_test = torch.Tensor(idx_test).long()


def steps(phase, Q):
    if phase=='train':
        recovered, x, q, y = HEDeFake(fts_node, GV, fts_edge, GE)
        p = target_distribution(Q.detach())

        loss1 = criteon(y, lbls)
        loss2 = F.binary_cross_entropy(recovered, H)
        loss_kl = F.kl_div(q.log(), p, reduction='batchmean')
        loss = loss1 + gamma1*loss2 + gamma2*loss_kl

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        schedular.step()
    else:
        with torch.no_grad():
            recovered, x, q, y = HEDeFake(fts_node, GV, fts_edge, GE)
            p = target_distribution(Q.detach())
            loss1 = criteon(y, lbls)
            loss2 = F.binary_cross_entropy(recovered, H)
            loss_kl = F.kl_div(q.log(), p, reduction='batchmean')
            loss = loss1 + gamma1*loss2 + gamma2*loss_kl

    return loss, x, y


def train_model(num_epochs=25, print_freq=50):
    best_hegnn_wts = copy.deepcopy(HEDeFake.state_dict())
    acc_best, loss_best, ep_best = 0, 10, -1

    kmeans = KMeans(n_clusters=2, n_init=20)
    y_pred = kmeans.fit_predict(fts_node.data.cpu().numpy())
    HEDeFake.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
    silhouette_avg = silhouette_score(fts_node.data.cpu().numpy(), y_pred)
    print(silhouette_avg)

    for epoch in range(num_epochs):
        printf('Epoch {:4d}'.format(epoch), style='bold')

        since = time.time()
        for phase in ['train', 'val']:
            if phase == 'train':
                HEDeFake.train()
                optimizer.zero_grad()
            else:
                HEDeFake.cpu().eval()

            idx = idx_train if phase == 'train' else idx_val

            if epoch % args_global.update_interval == 0:
                # update_interval
                recovered, x, Q, y = HEDeFake(fts_node, GV, fts_edge, GE)
                # q = Q.detach().data.cpu().numpy().argmax(1)

            loss, hn, he = steps(phase, Q)
            train_time = time.time() - since

            users_of_each_news = involved_users(H.cpu(), idx)
            agg_hn = Average(users_of_each_news, hn, num_news).cpu()
            preds = torch.add(he.cpu(), agg_hn, alpha=1)
            acc, pre, rec, f1 = evaluation(to_numpy(lbls[idx]), to_numpy(preds[idx]))

            if phase == 'train':
                printf('TRAIN (Ep avg): loss = {:.4f} \t accuracy = {:.4f}\tprecision = {:.4f}\trecall = {:.4f}\tF1  = {:.4f}\ttrain time = {:.4f} sec' \
                    .format(loss, acc, pre, rec, f1, train_time))
            if phase == 'val':
                printf('VALIDATION: loss = {:.4f} \t accuracy = {:.4f}\tprecision = {:.4f}\trecall = {:.4f}\tF1  = {:.4f}' \
                       .format(loss, acc, pre, rec, f1), style='yellow')
            if acc >= acc_best:
                acc_best, loss_best, ep_best = acc, loss, epoch
                best_hegnn_wts = copy.deepcopy(HEDeFake.state_dict())
                printf('  Saving model ...', style='yellow')
    printf("Optimization Finished! The best epoch is {}, ACC = {:.4f}".format(ep_best, acc_best), style="yellow")


# test
    HEDeFake.load_state_dict(best_hegnn_wts)
    hegnn_eval = HEDeFake
    printf('  Restoring model ...', style='yellow')

    idx = idx_test
    with torch.no_grad():
        _, hn, __, he = hegnn_eval(fts_node, GV, fts_edge, GE)
        users_of_each_news = involved_users(H.cpu(), idx)
        agg_hn = Average(users_of_each_news, hn, num_news)
        preds = torch.add(he, agg_hn, alpha=1)

    acc, pre, rec, f1 = evaluation(to_numpy(lbls[idx]), to_numpy(preds[idx]))
    printf('Test results: accuracy = {:.4f}\tprecision = {:.4f}\trecall = {:.4f}\tF1  = {:.4f}' \
           .format(acc, pre, rec, f1), style='red')


if __name__ == '__main__':
    HEDeFake = ComEDeFake(in_n=fts_node.shape[1], in_e=fts_edge.shape[1], n_out=fts_node.shape[1], n_class=n_class, n_hid=cfg['n_hid'], dropout=cfg['drop_out'])
    optimizer = optim.Adam(HEDeFake.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    schedular = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg['milestones'], gamma=cfg['gamma'])
    HEDeFake = HEDeFake.to(device)
    criteon = torch.nn.CrossEntropyLoss()
    gamma1, gamma2 = args_global.gamma1, args_global.gamma2

    # community-driven news classification
    model = train_model(cfg['max_epoch'], print_freq=cfg['print_freq'])

