"""Dual-head full-graph attention variational graph autoencoder.

A shared attention encoder extracts boundary context. Two independent attention
decoders reconstruct the unknown Dirichlet and Neumann quantities. The public
API remains compatible with the training and evaluation scripts.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax

@dataclass
class ModelOutput:
    """Container returned by the dual-head model."""
    pred_T_unknown: torch.Tensor
    pred_q_unknown: torch.Tensor
    mu: torch.Tensor | None = None
    logvar: torch.Tensor | None = None
    latent: torch.Tensor | None = None

def _select_num_heads(out_dim: int) -> int:
    """Choose a standard attention-head count that divides the feature width."""
    for heads in (8, 4, 2, 1):
        if out_dim % heads == 0:
            return heads
    return 1

class FullGraphAttentionConv(nn.Module):
    """Multi-head message passing with learned pairwise edge relations.

    Query-key compatibility provides the base attention score. A learned edge
    bias uses ``[source, target, target-source]`` features, while a second
    network gates each attention head. A residual projection preserves local
    node information.
    """
    def __init__(self, in_dim: int, out_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        if out_dim % num_heads != 0:
            raise ValueError("out_dim must be divisible by num_heads.")
        self.out_dim=out_dim; self.num_heads=num_heads; self.head_dim=out_dim//num_heads
        self.query_proj=nn.Linear(in_dim,out_dim,bias=False)
        self.key_proj=nn.Linear(in_dim,out_dim,bias=False)
        self.value_proj=nn.Linear(in_dim,out_dim,bias=False)
        edge_dim=3*in_dim; hidden=max(out_dim,32)
        self.edge_bias=nn.Sequential(nn.Linear(edge_dim,hidden),nn.GELU(),nn.Linear(hidden,num_heads))
        self.edge_gate=nn.Sequential(nn.Linear(edge_dim,hidden),nn.GELU(),nn.Linear(hidden,num_heads),nn.Sigmoid())
        self.output_proj=nn.Linear(out_dim,out_dim)
        self.residual_proj=nn.Identity() if in_dim==out_dim else nn.Linear(in_dim,out_dim)
        self.attention_dropout=nn.Dropout(dropout)

    def forward(self,x,edge_index):
        """Aggregate all non-self incoming messages for every target node."""
        n=x.size(0); residual=self.residual_proj(x)
        source,target=edge_index[0],edge_index[1]
        keep=source!=target; source=source[keep]; target=target[keep]
        if source.numel()==0: return residual
        q=self.query_proj(x).view(n,self.num_heads,self.head_dim)
        k=self.key_proj(x).view(n,self.num_heads,self.head_dim)
        v=self.value_proj(x).view(n,self.num_heads,self.head_dim)
        score=(q[target]*k[source]).sum(-1)/math.sqrt(self.head_dim)
        pair=torch.cat([x[source],x[target],x[target]-x[source]],dim=-1)
        score=score+self.edge_bias(pair)
        alpha=self.attention_dropout(softmax(score,index=target,num_nodes=n))
        gate=self.edge_gate(pair).unsqueeze(-1)
        msg=alpha.unsqueeze(-1)*gate*v[source]
        agg=torch.zeros((n,self.num_heads,self.head_dim),dtype=x.dtype,device=x.device)
        agg.index_add_(0,target,msg)
        return residual+self.output_proj(agg.reshape(n,self.out_dim))

class FullGraphAttentionBlock(nn.Module):
    """Attention followed by LayerNorm, GELU activation, and dropout."""
    def __init__(self,in_dim,out_dim,dropout=0.0):
        super().__init__()
        self.conv=FullGraphAttentionConv(in_dim,out_dim,_select_num_heads(out_dim),dropout)
        self.norm=nn.LayerNorm(out_dim); self.dropout=nn.Dropout(dropout)
    def forward(self,x,edge_index,edge_weight=None):
        return self.dropout(F.gelu(self.norm(self.conv(x,edge_index))))

class DualHeadGCNVAE(nn.Module):
    """Three-layer shared encoder and two independent three-layer decoders.

    The historical class name is retained to keep the surrounding scripts
    stable. Message passing is implemented by full-graph attention rather than
    GCN convolution.
    """
    def __init__(self,in_channels=9,hidden_channels=128,latent_channels=32,
                 decoder_channels=128,dropout=0.05,use_variational=True):
        super().__init__(); self.use_variational=use_variational
        self.enc_gcn1=FullGraphAttentionBlock(in_channels,hidden_channels,dropout)
        self.enc_gcn2=FullGraphAttentionBlock(hidden_channels,hidden_channels,dropout)
        self.enc_gcn3=FullGraphAttentionBlock(hidden_channels,hidden_channels,dropout)
        self.mu_head=nn.Sequential(nn.Linear(hidden_channels,hidden_channels),nn.GELU(),nn.Linear(hidden_channels,latent_channels))
        self.logvar_head=nn.Sequential(nn.Linear(hidden_channels,hidden_channels),nn.GELU(),nn.Linear(hidden_channels,latent_channels))
        d=in_channels+latent_channels
        self.t_gcn1=FullGraphAttentionBlock(d,decoder_channels,dropout)
        self.t_gcn2=FullGraphAttentionBlock(decoder_channels,decoder_channels,dropout)
        self.t_gcn3=FullGraphAttentionBlock(decoder_channels,decoder_channels,dropout)
        self.q_gcn1=FullGraphAttentionBlock(d,decoder_channels,dropout)
        self.q_gcn2=FullGraphAttentionBlock(decoder_channels,decoder_channels,dropout)
        self.q_gcn3=FullGraphAttentionBlock(decoder_channels,decoder_channels,dropout)
        self.t_out=nn.Sequential(nn.Linear(decoder_channels,decoder_channels//2),nn.GELU(),nn.Linear(decoder_channels//2,1))
        self.q_out=nn.Sequential(nn.Linear(decoder_channels,decoder_channels//2),nn.GELU(),nn.Linear(decoder_channels//2,1))

    def reparameterize(self,mu,logvar):
        if (not self.training) or (not self.use_variational): return mu
        return mu+torch.randn_like(mu)*torch.exp(0.5*logvar)
    def encode(self,x,edge_index,edge_weight=None):
        h=self.enc_gcn3(self.enc_gcn2(self.enc_gcn1(x,edge_index),edge_index),edge_index)
        mu=self.mu_head(h); logvar=self.logvar_head(h)
        return self.reparameterize(mu,logvar),mu,logvar
    def decode_t(self,xz,edge_index,edge_weight=None):
        h=self.t_gcn3(self.t_gcn2(self.t_gcn1(xz,edge_index),edge_index),edge_index)
        return self.t_out(h).squeeze(-1)
    def decode_q(self,xz,edge_index,edge_weight=None):
        h=self.q_gcn3(self.q_gcn2(self.q_gcn1(xz,edge_index),edge_index),edge_index)
        return self.q_out(h).squeeze(-1)
    def forward(self,data):
        z,mu,logvar=self.encode(data.x,data.edge_index,getattr(data,"edge_weight",None))
        xz=torch.cat([data.x,z],dim=-1)
        return ModelOutput(self.decode_t(xz,data.edge_index),self.decode_q(xz,data.edge_index),
                           mu if self.use_variational else None,
                           logvar if self.use_variational else None,z)

def assemble_full_fields(pred_T_unknown,pred_q_unknown,T_known,q_known,mask_T_known,mask_q_known):
    """Restore known boundary values and fill unknown entries with predictions."""
    return (mask_T_known*T_known+(1-mask_T_known)*pred_T_unknown,
            mask_q_known*q_known+(1-mask_q_known)*pred_q_unknown)

def supervised_unknown_loss(pred_T_unknown,pred_q_unknown,T_true,q_true,mask_T_known,mask_q_known):
    """Compute MSE only where the corresponding physical quantity is unknown."""
    mt=mask_q_known>0.5; mq=mask_T_known>0.5
    z=torch.tensor(0.0,device=pred_T_unknown.device,dtype=pred_T_unknown.dtype)
    lt=F.mse_loss(pred_T_unknown[mt],T_true[mt]) if torch.any(mt) else z
    lq=F.mse_loss(pred_q_unknown[mq],q_true[mq]) if torch.any(mq) else z
    return 0.5*(lt+lq),{"loss_T":lt,"loss_q":lq,"unknown_T_mask":mt,"unknown_q_mask":mq}

def kl_loss(mu,logvar,device=None):
    """Return the mean diagonal-Gaussian KL term."""
    if mu is None or logvar is None: return torch.tensor(0.0,device=device or "cpu")
    return -0.5*torch.mean(1+logvar-mu.pow(2)-logvar.exp())

def compute_losses(data,output,beta_kl=1e-5):
    """Compute training objective, scalar metrics, and reconstructed fields."""
    sup,aux=supervised_unknown_loss(output.pred_T_unknown,output.pred_q_unknown,data.T_true,data.q_true,data.mask_T_known,data.mask_q_known)
    kl=kl_loss(output.mu,output.logvar,device=sup.device); total=sup+beta_kl*kl
    T,q=assemble_full_fields(output.pred_T_unknown,output.pred_q_unknown,data.T_known,data.q_known,data.mask_T_known,data.mask_q_known)
    mt,mq=aux["unknown_T_mask"],aux["unknown_q_mask"]
    tmae=torch.mean(torch.abs(T[mt]-data.T_true[mt])) if torch.any(mt) else torch.tensor(0.,device=sup.device)
    qmae=torch.mean(torch.abs(q[mq]-data.q_true[mq])) if torch.any(mq) else torch.tensor(0.,device=sup.device)
    info={"loss_total":float(total.detach().cpu()),"loss_sup":float(sup.detach().cpu()),
          "loss_T":float(aux["loss_T"].detach().cpu()),"loss_q":float(aux["loss_q"].detach().cpu()),
          "loss_kl":float(kl.detach().cpu()),"T_unknown_mae":float(tmae.detach().cpu()),
          "q_unknown_mae":float(qmae.detach().cpu())}
    tensors={"T_full":T,"q_full":q,"unknown_T_mask":mt,"unknown_q_mask":mq,"latent":output.latent}
    return total,info,tensors
