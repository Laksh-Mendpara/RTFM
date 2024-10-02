import visdom
import torch
import torch.nn
import numpy as np
import config

    
def minmax_norm(act_map, min_val=None, max_val=None):
        if min_val is None or max_val is None: #if min/max value is not specified then it finds it, else it is skipped directly to delta part...
            relu=torch.nn.ReLU()
            max_val=relu(torch.max(act_map, dim=0)[0])
            min_val=relu(torch.min(act_map, dim=0)[0])
        delta=max_val-min_val
        delta[delta<=0]=1
        ret=(act_map-min_val)/delta #value belongs to range [0,1]
        ret[ret>1]=1
        ret[ret<0]=0
        return ret


def process_feat(feat, length):
    new_feat = np.zeros((length, feat.shape[1])).astype(np.float32)
    
    r = np.linspace(0, len(feat), length+1, dtype=np.int64)
    
    for i in range(length):
        if r[i] != r[i+1]:
            new_feat[i, :] = np.mean(feat[r[i]:r[i+1], :], axis=0)
        else:
            new_feat[i, :] = feat[r[i], :]
    
    return new_feat

def modelsize(model, input, type_size=4):
    para = sum([np.prod(list(p.size())) for p in model.parameters()])
    print('Model {} : params: {:4f}M'.format(model._get_name(), para * type_size / 1000 / 1000)) #gpu->utilisation check

    input_ = input.clone()
    input_.requires_grad_(requires_grad=False)

    mods = list(model.modules())
    out_sizes = []

    for i in range(1, len(mods)):
        m = mods[i]
        if isinstance(m, nn.ReLU):
            if m.inplace:
                continue
        out = m(input_)
        out_sizes.append(np.array(out.size()))
        input_ = out

    total_nums = 0
    for i in range(len(out_sizes)):
        s = out_sizes[i]
        nums = np.prod(np.array(s))
        total_nums += nums


    print('Model {} : intermedite variables: {:3f} M (without backward)'
          .format(model._get_name(), total_nums * type_size / 1000 / 1000))
    print('Model {} : intermedite variables: {:3f} M (with backward)'
          .format(model._get_name(), total_nums * type_size*2 / 1000 / 1000))


def save_checkpoint(model, optimizer, filename="checkpoints/my_checkpoint.pth.tar"):
    # print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
