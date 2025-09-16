import torch
from torch import nn
from Params import args
import torch.nn.functional as F
import math
import numpy as np
init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform

class Model(nn.Module):
   def __init__(self, image_embedding, text_embedding):
      super(Model, self).__init__()

      self.model_dict = {}
      self.model_dict['modal_consist'] = Model_modal_sh(image_embedding, text_embedding).cuda()
      self.model_dict['visual_compl'] = Model_visual_sp(image_embedding).cuda()
      self.model_dict['text_compl'] = Model_text_sp(text_embedding).cuda()

   def forward_current(self, current_model, adj, image_adj, text_adj):
      for model_name, model in self.model_dict.items():
         if model_name != current_model:
            continue
         usrEmbeds, itmEmbeds = model.forward_MM(adj, image_adj, text_adj)

      return usrEmbeds, itmEmbeds

   def forward_mask(self, mask_model, adj, image_adj, text_adj):
      usrEmbeds, itmEmbeds = None, None

      for model_name, model in self.model_dict.items():
         if model_name in mask_model:
            continue
         usrEmbed, itmEmbed = model.forward_MM(adj, image_adj, text_adj)

         if usrEmbeds is None or itmEmbeds is None:
            usrEmbeds = usrEmbed
            itmEmbeds = itmEmbed
         else:
            usrEmbeds = torch.cat([usrEmbeds, usrEmbed], dim=-1)
            itmEmbeds = torch.cat([itmEmbeds, itmEmbed], dim=-1)

      return usrEmbeds, itmEmbeds


class Model_modal_sh(nn.Module):
   def __init__(self, image_embedding, text_embedding):
      super(Model_modal_sh, self).__init__()

      self.uEmbeds = nn.Parameter(init(torch.empty(args.user, args.latdim)))
      self.iEmbeds = nn.Parameter(init(torch.empty(args.item, args.latdim)))

      self.gcnLayers = nn.Sequential(*[GCNLayer() for i in range(args.gnn_layer)])

      self.edgeDropper = SpAdjDropEdge(args.keepRate)

      if args.trans == 1:
         self.image_trans_sh = nn.Linear(args.image_feat_dim, args.latdim)
         self.text_trans_sh = nn.Linear(args.text_feat_dim, args.latdim)
      elif args.trans == 0:
         self.image_trans_sh = nn.Parameter(init(torch.empty(size=(args.image_feat_dim, args.latdim))))
         self.text_trans_sh = nn.Parameter(init(torch.empty(size=(args.text_feat_dim, args.latdim))))

      self.image_embedding = image_embedding
      self.text_embedding = text_embedding

      self.dropout = nn.Dropout(p=0.1)

      self.leakyrelu = nn.LeakyReLU(0.2)

   def getItemEmbeds(self):
      return self.iEmbeds

   def getUserEmbeds(self):
      return self.uEmbeds

   def getImageFeats(self):
      if args.trans == 0 or args.trans == 2:
         image_feats_sh = self.leakyrelu(torch.mm(self.image_embedding, self.image_trans_sh))
         return image_feats_sh
      else:
         return self.image_trans_sh(self.image_embedding)

   def getTextFeats(self):
      if args.trans == 0 or args.trans == 2:
         text_feats_sh = self.leakyrelu(torch.mm(self.text_embedding, self.text_trans_sh))
         return text_feats_sh
      else:
         return self.text_trans_sh(self.text_embedding)

   def forward_MM(self, adj, image_adj, text_adj):
      if args.trans == 0:
         image_feats_sh = self.leakyrelu(torch.mm(self.image_embedding, self.image_trans_sh))
         text_feats_sh = self.leakyrelu(torch.mm(self.text_embedding, self.text_trans_sh))
      elif args.trans == 1:
         image_feats_sh = self.image_trans_sh(self.image_embedding)
         text_feats_sh = self.text_trans_sh(self.text_embedding)


      embedsImageAdj_sh = torch.concat([self.uEmbeds, self.iEmbeds])
      embedsImageAdj_sh = torch.spmm(image_adj, embedsImageAdj_sh)
      embedsImage_sh = torch.concat([self.uEmbeds, F.normalize(image_feats_sh)])
      embedsImage_sh = torch.spmm(adj, embedsImage_sh)
      embedsImage_sh_ = torch.concat([embedsImage_sh[:args.user], self.iEmbeds])
      embedsImage_sh_ = torch.spmm(adj, embedsImage_sh_)
      embedsImage_sh += embedsImage_sh_

      embedsImage_sh += args.ris_adj_lambda * embedsImageAdj_sh

      embedsLst = [embedsImage_sh]
      for gcn in self.gcnLayers:
         embeds_visual_sh = gcn(adj, embedsLst[-1])
         embedsLst.append(embeds_visual_sh)
      embeds_visual_sh = sum(embedsLst)

      embedsTextAdj_sh = torch.concat([self.uEmbeds, self.iEmbeds])
      embedsTextAdj_sh = torch.spmm(text_adj, embedsTextAdj_sh)
      embedsText_sh = torch.concat([self.uEmbeds, F.normalize(text_feats_sh)])
      embedsText_sh = torch.spmm(adj, embedsText_sh)
      embedsText_sh_ = torch.concat([embedsText_sh[:args.user], self.iEmbeds])
      embedsText_sh_ = torch.spmm(adj, embedsText_sh_)
      embedsText_sh += embedsText_sh_

      embedsText_sh += args.ris_adj_lambda * embedsTextAdj_sh

      embedsLst = [embedsText_sh]
      for gcn in self.gcnLayers:
         embeds_text_sh = gcn(adj, embedsLst[-1])
         embedsLst.append(embeds_text_sh)
      embeds_text_sh = sum(embedsLst)

      embeds = torch.cat([embeds_visual_sh,embeds_text_sh],dim=-1)

      return embeds[:args.user], embeds[args.user:]

   def reg_loss(self):
      ret = 0
      ret += self.uEmbeds.norm(2).square()
      ret += self.iEmbeds.norm(2).square()
      return ret




class Model_visual_sp(nn.Module):
   def __init__(self, image_embedding):
      super(Model_visual_sp, self).__init__()

      self.uEmbeds_vsp = nn.Parameter(init(torch.empty(args.user, args.latdim)))
      self.iEmbeds_vsp = nn.Parameter(init(torch.empty(args.item, args.latdim)))

      self.gcnLayers = nn.Sequential(*[GCNLayer() for i in range(args.gnn_layer)])

      self.edgeDropper = SpAdjDropEdge(args.keepRate)

      if args.trans == 1:
         self.image_trans_sp = nn.Linear(args.image_feat_dim, args.latdim)
      elif args.trans == 0:
         self.image_trans_sp = nn.Parameter(init(torch.empty(size=(args.image_feat_dim, args.latdim))))

      self.image_embedding = image_embedding

      self.dropout = nn.Dropout(p=0.1)

      self.leakyrelu = nn.LeakyReLU(0.2)

   def getItemEmbeds(self):
      return self.iEmbeds_vsp

   def getUserEmbeds(self):
      return self.uEmbeds_vsp

   def getImageFeats(self):
      if args.trans == 0 or args.trans == 2:
         image_feats_sp = self.leakyrelu(torch.mm(self.image_embedding, self.image_trans_sp))
         return image_feats_sp
      else:
         return self.image_trans_sp(self.image_embedding)

   def forward_MM(self, adj, image_adj, text_adj):
      if args.trans == 0:
         image_feats_sp = self.leakyrelu(torch.mm(self.image_embedding, self.image_trans_sp))
      elif args.trans == 1:
         image_feats_sp = self.image_trans_sp(self.image_embedding)

      embedsImageAdj_sp = torch.concat([self.uEmbeds_vsp, self.iEmbeds_vsp])
      embedsImageAdj_sp = torch.spmm(image_adj, embedsImageAdj_sp)
      embedsImage_sp = torch.concat([self.uEmbeds_vsp, F.normalize(image_feats_sp)])
      embedsImage_sp = torch.spmm(adj, embedsImage_sp)
      embedsImage_sp_ = torch.concat([embedsImage_sp[:args.user], self.iEmbeds_vsp])
      embedsImage_sp_ = torch.spmm(adj, embedsImage_sp_)
      embedsImage_sp += embedsImage_sp_


      embedsImage_sp += args.ris_adj_lambda * embedsImageAdj_sp

      embedsLst = [embedsImage_sp]
      for gcn in self.gcnLayers:
         embeds_visual_sp = gcn(adj, embedsLst[-1])
         embedsLst.append(embeds_visual_sp)
      embeds_visual_sp = sum(embedsLst)

      return embeds_visual_sp[:args.user], embeds_visual_sp[args.user:]


   def reg_loss(self):
      ret = 0
      ret += self.uEmbeds_vsp.norm(2).square()
      ret += self.iEmbeds_vsp.norm(2).square()
      return ret


class Model_text_sp(nn.Module):
   def __init__(self, text_embedding):
      super(Model_text_sp, self).__init__()

      self.uEmbeds_tsp = nn.Parameter(init(torch.empty(args.user, args.latdim)))
      self.iEmbeds_tsp = nn.Parameter(init(torch.empty(args.item, args.latdim)))

      self.gcnLayers = nn.Sequential(*[GCNLayer() for i in range(args.gnn_layer)])

      self.edgeDropper = SpAdjDropEdge(args.keepRate)

      if args.trans == 1:
         self.text_trans_sp = nn.Linear(args.text_feat_dim, args.latdim)
      elif args.trans == 0:
         self.text_trans_sp = nn.Parameter(init(torch.empty(size=(args.text_feat_dim, args.latdim))))

      self.text_embedding = text_embedding

      self.dropout = nn.Dropout(p=0.1)

      self.leakyrelu = nn.LeakyReLU(0.2)

   def getItemEmbeds(self):
      return self.iEmbeds_tsp

   def getUserEmbeds(self):
      return self.uEmbeds_tsp

   def getTextFeats(self):
      if args.trans == 0 or args.trans == 2:
         text_feats_sp = self.leakyrelu(torch.mm(self.text_embedding, self.text_trans_sp))
         return text_feats_sp
      else:
         return self.text_trans_sp(self.text_embedding)

   def forward_MM(self, adj, image_adj, text_adj):
      if args.trans == 0:
         text_feats_sp = self.leakyrelu(torch.mm(self.text_embedding, self.text_trans_sp))
      elif args.trans == 1:
         text_feats_sp = self.text_trans_sp(self.text_embedding)

      embedsTextAdj_sp = torch.concat([self.uEmbeds_tsp, self.iEmbeds_tsp])
      embedsTextAdj_sp = torch.spmm(text_adj, embedsTextAdj_sp)
      embedsText_sp = torch.concat([self.uEmbeds_tsp, F.normalize(text_feats_sp)])
      embedsText_sp = torch.spmm(adj, embedsText_sp)
      embedsText_sp_ = torch.concat([embedsText_sp[:args.user], self.iEmbeds_tsp])
      embedsText_sp_ = torch.spmm(adj, embedsText_sp_)
      embedsText_sp += embedsText_sp_

      embedsText_sp += args.ris_adj_lambda * embedsTextAdj_sp

      embedsLst = [embedsText_sp]
      for gcn in self.gcnLayers:
         embeds_text_sp = gcn(adj, embedsLst[-1])
         embedsLst.append(embeds_text_sp)
      embeds_text_sp = sum(embedsLst)

      return embeds_text_sp[:args.user], embeds_text_sp[args.user:]


   def reg_loss(self):
      ret = 0
      ret += self.uEmbeds_tsp.norm(2).square()
      ret += self.iEmbeds_tsp.norm(2).square()
      return ret

class Denoise(nn.Module):
   def __init__(self, in_dims, out_dims, emb_size, norm=False, dropout=0.5):
      super(Denoise, self).__init__()
      self.in_dims = in_dims
      self.out_dims = out_dims
      self.time_emb_dim = emb_size
      self.norm = norm

      self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)

      in_dims_temp = [self.in_dims[0] + self.time_emb_dim] + self.in_dims[1:]

      out_dims_temp = self.out_dims

      self.in_layers = nn.ModuleList(
         [nn.Linear(d_in, d_out) for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:])])
      self.out_layers = nn.ModuleList(
         [nn.Linear(d_in, d_out) for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:])])

      self.drop = nn.Dropout(dropout)
      self.init_weights()

   def init_weights(self):
      for layer in self.in_layers:
         size = layer.weight.size()
         std = np.sqrt(2.0 / (size[0] + size[1]))
         layer.weight.data.normal_(0.0, std)
         layer.bias.data.normal_(0.0, 0.001)

      for layer in self.out_layers:
         size = layer.weight.size()
         std = np.sqrt(2.0 / (size[0] + size[1]))
         layer.weight.data.normal_(0.0, std)
         layer.bias.data.normal_(0.0, 0.001)

      size = self.emb_layer.weight.size()
      std = np.sqrt(2.0 / (size[0] + size[1]))
      self.emb_layer.weight.data.normal_(0.0, std)
      self.emb_layer.bias.data.normal_(0.0, 0.001)

   def forward(self, x, timesteps, mess_dropout=True):
      freqs = torch.exp(-math.log(10000) * torch.arange(start=0, end=self.time_emb_dim // 2, dtype=torch.float32) / (
                 self.time_emb_dim // 2)).cuda()
      temp = timesteps[:, None].float() * freqs[None]
      time_emb = torch.cat([torch.cos(temp), torch.sin(temp)], dim=-1)
      if self.time_emb_dim % 2:
         time_emb = torch.cat([time_emb, torch.zeros_like(time_emb[:, :1])], dim=-1)
      emb = self.emb_layer(time_emb)
      if self.norm:
         x = F.normalize(x)
      if mess_dropout:
         x = self.drop(x)
      h = torch.cat([x, emb], dim=-1)
      for i, layer in enumerate(self.in_layers):
         h = layer(h)
         h = torch.tanh(h)
      for i, layer in enumerate(self.out_layers):
         h = layer(h)
         if i != len(self.out_layers) - 1:
            h = torch.tanh(h)

      return h


class GaussianDiffusion(nn.Module):
   def __init__(self, noise_scale, noise_min, noise_max, steps, beta_fixed=True):
      super(GaussianDiffusion, self).__init__()

      self.noise_scale = noise_scale
      self.noise_min = noise_min
      self.noise_max = noise_max
      self.steps = steps

      if noise_scale != 0:
         self.betas = torch.tensor(self.get_betas(), dtype=torch.float64).cuda()
         if beta_fixed:
            self.betas[0] = 0.0001

         self.calculate_for_diffusion()

   def get_betas(self):
      start = self.noise_scale * self.noise_min
      end = self.noise_scale * self.noise_max
      variance = np.linspace(start, end, self.steps, dtype=np.float64)
      alpha_bar = 1 - variance
      betas = []
      betas.append(1 - alpha_bar[0])
      for i in range(1, self.steps):
         betas.append(min(1 - alpha_bar[i] / alpha_bar[i - 1], 0.999))
      return np.array(betas)

   def calculate_for_diffusion(self):
      alphas = 1.0 - self.betas
      self.alphas_cumprod = torch.cumprod(alphas, axis=0).cuda()
      self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]).cuda(), self.alphas_cumprod[:-1]]).cuda()
      self.alphas_cumprod_next = torch.cat([self.alphas_cumprod[1:], torch.tensor([0.0]).cuda()]).cuda()

      self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
      self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
      self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
      self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
      self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

      self.posterior_variance = (
              self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
      )
      self.posterior_log_variance_clipped = torch.log(
         torch.cat([self.posterior_variance[1].unsqueeze(0), self.posterior_variance[1:]]))
      self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
      self.posterior_mean_coef2 = ((1.0 - self.alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - self.alphas_cumprod))

   def p_sample(self, model, x_start, steps, sampling_noise=False):
      if steps == 0:
         x_t = x_start
      else:
         t = torch.tensor([steps - 1] * x_start.shape[0]).cuda()
         x_t = self.q_sample(x_start, t)

      indices = list(range(self.steps))[::-1]

      for i in indices:
         t = torch.tensor([i] * x_t.shape[0]).cuda()
         model_mean, model_log_variance = self.p_mean_variance(model, x_t, t)
         if sampling_noise:
            noise = torch.randn_like(x_t)
            nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1))))
            x_t = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
         else:
            x_t = model_mean
      return x_t

   def q_sample(self, x_start, t, noise=None):
      if noise is None:
         noise = torch.randn_like(x_start)
      return self._extract_into_tensor(self.sqrt_alphas_cumprod, t,
                                       x_start.shape) * x_start + self._extract_into_tensor(
         self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise

   def _extract_into_tensor(self, arr, timesteps, broadcast_shape):
      arr = arr.cuda()
      res = arr[timesteps].float()
      while len(res.shape) < len(broadcast_shape):
         res = res[..., None]
      return res.expand(broadcast_shape)

   def p_mean_variance(self, model, x, t):
      model_output = model(x, t, False)

      model_variance = self.posterior_variance
      model_log_variance = self.posterior_log_variance_clipped

      model_variance = self._extract_into_tensor(model_variance, t, x.shape)
      model_log_variance = self._extract_into_tensor(model_log_variance, t, x.shape)

      model_mean = (self._extract_into_tensor(self.posterior_mean_coef1, t,
                                              x.shape) * model_output + self._extract_into_tensor(
         self.posterior_mean_coef2, t, x.shape) * x)

      return model_mean, model_log_variance

   def training_losses(self, model, x_start, itmEmbeds, batch_index, model_feats):
      batch_size = x_start.size(0)

      ts = torch.randint(0, self.steps, (batch_size,)).long().cuda()
      noise = torch.randn_like(x_start)
      if self.noise_scale != 0:
         x_t = self.q_sample(x_start, ts, noise)
      else:
         x_t = x_start

      model_output = model(x_t, ts)

      mse = self.mean_flat((x_start - model_output) ** 2)

      weight = self.SNR(ts - 1) - self.SNR(ts)
      weight = torch.where((ts == 0), 1.0, weight)

      diff_loss = weight * mse

      usr_model_embeds = torch.mm(model_output, model_feats)
      usr_id_embeds = torch.mm(x_start, itmEmbeds)

      gc_loss = self.mean_flat((usr_model_embeds - usr_id_embeds) ** 2)

      return diff_loss, gc_loss

   def mean_flat(self, tensor):
      return tensor.mean(dim=list(range(1, len(tensor.shape))))

   def SNR(self, t):
      self.alphas_cumprod = self.alphas_cumprod.cuda()
      return self.alphas_cumprod[t] / (1 - self.alphas_cumprod[t])

class GCNLayer(nn.Module):
   def __init__(self):
      super(GCNLayer, self).__init__()

   def forward(self, adj, embeds):
      return torch.spmm(adj, embeds)


class SpAdjDropEdge(nn.Module):
   def __init__(self, keepRate):
      super(SpAdjDropEdge, self).__init__()
      self.keepRate = keepRate

   def forward(self, adj):
      vals = adj._values()
      idxs = adj._indices()
      edgeNum = vals.size()
      mask = ((torch.rand(edgeNum) + self.keepRate).floor()).type(torch.bool)

      newVals = vals[mask] / self.keepRate
      newIdxs = idxs[:, mask]

      return torch.sparse.FloatTensor(newIdxs, newVals, adj.shape)

