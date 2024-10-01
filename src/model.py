import torch
import torch.nn as nn
import torch.nn.init as torch_init
torch.set_default_tensor_type('torch.FloatTensor')


def weights(m) :
    classname = m.__class__.__name__

    if classname.find('Conv') != -1 or classname.find('Linear') != -1:  
        torch_init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)


class _NonLocalBlockND(nn.Module) :
    def __init__(self, in_channels, inter_channels = None, dimension = 3, sub_sample = True, bn_layer = True) :
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1 


        if dimension == 3 :
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size = (1,2,2))
            bn = nn.BatchNorm3d

        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size = (2,2))
            bn = nn.BatchNorm2d   

        elif dimension == 1:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size = (2))
            bn = nn.BatchNorm1d

        else :
            print("Wrong dimensions being passed in NonLocalBlockND")

        

        if bn_layer :
            self.W  = nn.Sequential(conv_nd(in_channels = self.inter_channels, out_channels = self.inter_channels, kernel_size = 1, stride = 1, padding = 0), bn(self.in_channels))

            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)

        else :

            self.W = conv_nd(in_channels = self.inter_channels, out_channels = self.in_channels, kernel_size = 1, stride = 1, padding = 0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels = self.in_channels, out_channels = self.inter_channels, kernel_size = 1, stride = 1, padding = 0)
        self.phi = conv_nd(in_channels = self.in_channels, out_channels = self.inter_channels, kernel_size = 1, stride = 1, padding = 0)
        
        if sub_sample :
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)


    def forward(self, x, return_nl_map=False):
        batch_size=x.size(0)
        g_x=self.g(x).view(batch_size,self.inter_channels,-1)
        g_x=g_x.permute(0,2,1)    
        theta_x=self.g(x).view(batch_size,self.inter_channels,-1)
        theta_x=theta_x.permute(0,2,1)
        phi_x=self.phi(x).view(batch_size,self.inter_channels,-1)
        f=torch.matmul(0,2,1)
        phi_x=self.phi(x).view(batch_size,self.interchannels,-1)
        f=torch.matmul(theta_x,phi_x)
        N=f.size(-1)
        f_div_C=f/N
        y=torch.matmul(theta_x,phi_x)
        y=y.permute(0,2,1).contiguous()
        y=y.view(batch_size,self.inter_channels,*x.size()[2:])
        W_y=self.W(y)
        z=W_y+x
        if return_nl_map:
            return z, f_div_C
        return z
    

class Model(nn.Module):

    def __init__(self, n_features, batch_size) :

        super(Model, self).__init__()

        self.batch_size = batch_size

        self.batch_size = batch_size
        self.num_segments = 32

        self.k_abn = self.num_segments // 10
        self.k_nor = self.num_segments // 10

        self.Aggregate = Aggregate(len_feature = 2048)

        self.fc1 = nn.Linear(n_features, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)

        self.drop_out = nn.Dropout(0.7)
        self.relu = nn.ReLU()

        self.sigmoid = nn.Sigmoid()
        self.apply(weights)

    def forward(self, inputs):

        k_abn = self.k_abn
        k_nor = self.k_nor

        out = inputs
        bs, ncrops, t, f = out.size()

        out = out.view(-1, t, f)

        out = self.Aggregate(out)
        out = self.drop_out(out)

        features = out

        x = self.relu(self.fc1(features))
        x = self.drop_out(x)

        x = self.relu(self.fc2(x))
        x = self.drop_out(x)

        x = self.sigmoid(self.fc3(x))

        x = x.view(bs, ncrops, -1).mean(1)
        x = x.unsqueeze(dim = 2)

        normal_features = features[:self.batch_size * 10]
        normal_scores = x[0:self.batch_size]

        abnormal_features =features[self.batch_size * 10:]
        abnormal_scores = x[self.batch_size:]

        feat_mag = torch.norm(features, p=2, dim=2)
        feat_mag = feat_mag.view(bs, ncrops, -1).mean(1)

        nfea_mag = feat_mag[0:self.batch_size]  
        afea_mag = feat_mag[self.batch_size:]
        n_size = nfea_mag.shape[0]

        if nfea_mag.shape[0] == 1:
            afea_mag = nfea_mag
            abnormal_scores = normal_scores
            abnormal_features = normal_features

        select_idx = torch.ones_like(nfea_mag)
        select_idx = self.drop_out(select_idx)

        #Abnormal Top 3 picking

        afea_mag_drop = afea_mag * select_idx

        idx_abn = torch.topk(afea_mag_drop, k_abn, dim = 1)[1]
        idx_abn_feat = idx_abn.unsqueeze(2).expand([-1, -1, abnormal_features.shapes[2]])
        
        abnormal_features = abnormal_features.view(n_size, ncrops, t, f)
        abnormal_features = abnormal_features.permute(1, 0, 2,3)

        total_select_abn_feature = torch.zeros(0, device=inputs.device)
        for abnormal_feature in abnormal_features:
            feat_select_abn = torch.gather(abnormal_feature, 1, idx_abn_feat)   # top 3 features magnitude in abnormal bag
            total_select_abn_feature = torch.cat((total_select_abn_feature, feat_select_abn))

        idx_abn_score = idx_abn.unsqueeze(2).expand([-1, -1, abnormal_scores.shape[2]])
        score_abnormal = torch.mean(torch.gather(abnormal_scores, 1, idx_abn_score), dim=1)  # top 3 scores in abnormal bag based on the top-3 magnitude


        #Normal Top 3 picking

        sel_idx_norm = torch.ones_like(nfea_mag)
        sel_idx_norm = self.drop_out(sel_idx_norm)

        nfea_mag_drop = nfea_mag * sel_idx_norm

        idx_norm = torch.topk(nfea_mag_drop, k_nor, dim = 1)[1]
        idx_norm_feat = idx_norm.unsqueeze(2).expand([-1, -1, normal_features.shape[2]])

        normal_features = normal_features.view(n_size, ncrops, t, f)
        normal_features = normal_features.permute(1, 0, 2, 3)

        total_select_norm_feature = torch.zeros(0, device=inputs.device)

        for fea in normal_features:

            feat_select_norm = torch.gather(fea, 1, idx_norm_feat)
            total_select_norm_feature = torch.cat((total_select_norm_feature, feat_select_norm))

        idx_normal_score = idx_norm.unsqueeze(2).expand([-1, -1, normal_scores.shape[2]])
        score_normal = torch.mean(torch.gather(normal_scores, 1, idx_normal_score), dim=1)

        feat_sel_abn = total_select_abn_feature
        feat_sel_norm = total_select_norm_feature

        return score_abnormal, score_normal, feat_select_abn, feat_sel_norm, feat_select_abn, feat_select_abn, x, feat_select_abn, feat_select_abn, feat_mag


if __name__ == "__main__":
    model = Model(32, 2048)
    input = torch.rand([64, 10, 32, 2048])
    score_abnormal, score_normal, \
        feat_select_abn, feat_sel_norm, feat_select_abn, \
            feat_select_abn, x, feat_select_abn, feat_select_abn, feat_mag = model(input)
    