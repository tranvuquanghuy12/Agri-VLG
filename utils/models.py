import torch
from torch import nn
import json
import pickle
from utils.features import prompt_sampler, get_text_features
from utils.extras import get_engine
from utils.datasets.dataset_utils import NUM_CLASSES_DICT
from utils import features
try:
    from torch_geometric.nn import GATConv
except ImportError:
    GATConv = None


class GATLayer(nn.Module):
    def __init__(self, in_channels, out_channels, heads=8, dropout=0.6):
        super(GATLayer, self).__init__()
        if GATConv is None:
            pass
        self.gat = GATConv(in_channels, out_channels // heads, heads=heads, dropout=dropout)

    def forward(self, x, return_attention=False):
        batch_size = x.size(0)
        edge_index = torch.ones((batch_size, batch_size), device=x.device).nonzero().t().contiguous()
        
        if return_attention:
            feature_interaction, (edge_index, alpha) = self.gat(x, edge_index, return_attention_weights=True)
            return x + feature_interaction, (edge_index, alpha)
        
        feature_interaction = self.gat(x, edge_index)
        return x + feature_interaction

class SWATModelWithGAT(nn.Module):
    def __init__(self, clip_model, use_gat=True):
        super().__init__()
        self.clip = clip_model
        self.use_gat = use_gat
        
        if hasattr(clip_model, 'visual') and hasattr(clip_model.visual, 'output_dim'):
            embed_dim = clip_model.visual.output_dim
        elif hasattr(clip_model, 'output_dim'):
            embed_dim = clip_model.output_dim
        else:
            embed_dim = 512
            
        self.gat = GATLayer(embed_dim, embed_dim)
        
    def encode_image(self, image, source=None, return_attention=False):
        features = self.clip.encode_image(image)
        if self.use_gat:
            if return_attention:
                features, attn = self.gat(features, return_attention=True)
                return features, attn
            features = self.gat(features)
        return features

    def encode_text(self, text):
        return self.clip.encode_text(text)

    @property
    def logit_scale(self):
        return self.clip.logit_scale

    @property
    def visual(self):
        return self.clip.visual

    @property
    def transformer(self):
        return self.clip.transformer

    @property
    def token_embedding(self):
        return self.clip.token_embedding

    @property
    def positional_embedding(self):
        return self.clip.positional_embedding

    @property
    def ln_final(self):
        return self.clip.ln_final

    @property
    def text_projection(self):
        return self.clip.text_projection

    def forward(self, image, source=None, text=None, return_attention=False):
        if text is not None:
            image_features = self.encode_image(image, source=source, return_attention=return_attention)
            if return_attention:
                image_features, attn = image_features
            text_features = self.encode_text(text)
            if return_attention:
                return (image_features, text_features, self.logit_scale.exp()), attn
            return image_features, text_features, self.logit_scale.exp()
        return self.encode_image(image, source=source, return_attention=return_attention)


def set_model(args, logger):
    model, preprocess, tokenizer = get_engine(model_cfg=args.model_cfg, device=args.device)
    logger.info(f'Loaded model: {args.model_cfg}')
    model = SWATModelWithGAT(model, use_gat=args.use_gat)
    model.to(args.device)
    logger.info("Integrated GAT (Graph Attention Network) for Stage-Wise Finetuning.")
    return model, preprocess, tokenizer


def set_classifier(args, prompt_tensors, logger):
    if args.method == "dataset-cls":
        num_class = 2
        classifier_head = MyLinear(inp_dim=512, num_classes=num_class, bias=False)    
    elif args.cls_init == 'REAL-Prompt' or args.cls_init == 'REAL-Linear' or args.cls_init == 'text':
        weights = features.prompt_sampler(prompt_tensors, sample_by='mean')
        classifier_head = MyLinear(weights=weights)
    elif args.cls_init == 'random':
        num_class = NUM_CLASSES_DICT[args.dataset]
        classifier_head = MyLinear(inp_dim=512, num_classes=num_class, bias=False)    
    else:
        raise NotImplementedError(f'Classifier head {args.cls_init} not implemented.')
    classifier_head.to(args.device)
    return classifier_head


class MyLinear(nn.Module):
    def __init__(self, weights=None, inp_dim=512, num_classes=810, bias = False):
        super(MyLinear, self).__init__()
        if torch.is_tensor(weights):
            self.linear = nn.Linear(weights.shape[1], weights.shape[0], bias=bias)
            with torch.no_grad():
                self.linear.weight.copy_(weights)
            self.num_classes = weights.shape[0]
        else:
            self.linear = nn.Linear(inp_dim, num_classes, bias=bias)
            self.num_classes = num_classes
    
    def forward(self, x):
        x = self.linear(x)
        return x
    
    def update_weights(self, weights): 
        with torch.no_grad():
            self.linear.weight.copy_(weights)


def build_classifier_head(args, model, text_prompts, tokenizer):
    updated_prompt_tensors = get_text_features(model, text_prompts, tokenizer, 'encode')
    weights = prompt_sampler(updated_prompt_tensors, sample_by='mean')
    new_head = MyLinear(weights=weights, bias=False) 
    new_head.to(args.device)
    return new_head


def save_model_ckpt(args, best_records, model, classifier_head, optimizer, scheduler, logit_scale,
                    val_acc=-1, epoch=-1, num_iter=-1):
    model_path = f'{args.ckpt_path}/model_bs{args.bsz}_lr-cls{args.lr_classifier}_lr-bkb{args.lr_backbone}_wd{args.wd}_epoch_{epoch}_iter_{num_iter}.pth'
    state = {}
    state['best_val_acc'] = best_records.get('best_val_acc', -1)
    state['best_epoch'] = best_records.get('best_epoch', -1)
    state['best_iter'] = best_records.get('best_iter', -1)
    state['val_acc'] = val_acc
    state['epoch'] = epoch
    state['num_iter'] = num_iter
    if not args.freeze_visual:
        state['clip'] = model.state_dict()
    state['head'] = classifier_head.state_dict()
    state['optimizer'] = optimizer.state_dict()
    state['scheduler'] = scheduler.state_dict()
    state['logit_scale'] = logit_scale
    torch.save(state, model_path)
    return model_path

         
def save_best_model(args, best_records, best_model, best_head, best_logit_scale,
                    test_acc, best_tau, best_tau_test_acc, wsft_test_acc,
                    best_tau_head, wsft_backbone, wsft_head, stage=1):
    best_epoch = best_records['best_epoch']
    best_iter = best_records['best_iter']
    model_path = f'{args.output_dir}/stage{stage}_model_best-epoch_{best_epoch}_best.pth'
    save_path = f'{args.output_dir}/stage{stage}_val_scores_best.json'
    with open(save_path, 'w') as f:
        json.dump(best_records['best_scores'], f, indent=4)
    save_path = f'{args.output_dir}/stage{stage}_val_confusion_matrix_best.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(best_records['best_confusion_matrix'], f)
    state = {}
    state['best_val_acc'] = best_records['best_val_acc']
    state['best_epoch'] = best_records['best_epoch']
    state['best_iter'] = best_records['best_iter']
    state['clip'] = best_model.state_dict()
    state['head'] = best_head.state_dict()
    state['logit_scale'] = best_logit_scale
    state['test_acc'] = round(test_acc, 3)
    state['best_tau'] = best_tau
    state['best_tau_test_acc'] = round(best_tau_test_acc, 3)
    state['wsft_test_acc'] = round(wsft_test_acc,3)
    state['best_tau_head'] = best_tau_head.state_dict() if best_tau_head is not None else None
    state['wsft_head'] = wsft_head.state_dict() if wsft_head is not None else None
    state['wsft_backbone'] = wsft_backbone.state_dict() if wsft_backbone is not None else None
    torch.save(state, model_path)
    return model_path


def save_test_scores(scores, confusion_matrix, output_dir, tag, stage=1):
    save_path = f'{output_dir}/stage{stage}_{tag}_scores.json'
    with open(save_path, 'w') as f:
        json.dump(scores, f, indent=4)
    save_path = f'{output_dir}/stage{stage}_{tag}_confusion_matrix.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(confusion_matrix, f)    

def save_head_weights(classifier_head, output_dir, tag):
    save_path = f'{output_dir}/{tag}_head_weights.pth'
    torch.save(classifier_head.state_dict(), save_path)
