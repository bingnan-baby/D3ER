import torch
import Utils.TimeLogger as logger
from Utils.TimeLogger import log
from Params import args
from Model_d3er import Model, GaussianDiffusion, Denoise
from DataHandler import DataHandler
import numpy as np
from Utils.Utils import *
import os
import scipy.sparse as sp
import random
from scipy.sparse import coo_matrix
import torch.nn as nn
import matplotlib.pyplot as plt
from swd import sliced_wasserstein_distance


# torch.autograd.set_detect_anomaly(True)

class Coach:
   def __init__(self, handler):
      self.handler = handler

      print('USER', args.user, 'ITEM', args.item)
      print('NUM OF INTERACTIONS', self.handler.trnLoader.dataset.__len__())
      self.metrics = dict()
      mets = ['Loss', 'preLoss', 'Recall', 'NDCG']
      for met in mets:
         self.metrics['Train' + met] = list()
         self.metrics['Test' + met] = list()

   def makePrint(self, name, ep, epoch, reses, save):
      ret = 'Epoch %d/%d, %s: ' % (
         ep, epoch, name)
      for metric in reses:
         val = reses[metric]
         ret += '%s = %.4f, ' % (metric, val)
         tem = name + metric
         if save and tem in self.metrics:
            self.metrics[tem].append(val)
      ret = ret[:-2] + '  '
      return ret

   def run_boost(self):
      tstFlag = 1
      self.prepareModel()
      log('Model Prepared')

      self.model_name_list = ['modal_consist', 'text_compl', 'visual_compl']

      recallMax = 0
      ndcgMax = 0
      precisionMax = 0
      bestEpoch = 0

      log('Model Initialized')

      for ep in range(0, args.warmup_epoch):
         eploss = self.trainEpoch_pretrain()
         log(self.makePrint('Train', ep, args.warmup_epoch, eploss, save=0))

      for stage in range(0, args.stage):
         trained_model_name = self.model_name_list[stage % 3]
         print(f'Stage {stage} begins! Trained modal : {trained_model_name}')

         trained_model = Model(self.handler.image_feats.detach(), self.handler.text_feats.detach()).cuda()
         for i in self.model_name_list:
            trained_model.model_dict[i].load_state_dict(self.model_ensemble.model_dict[i].state_dict())

         # phase 1
         opt = torch.optim.Adam(trained_model.model_dict[trained_model_name].parameters(), lr=args.lr, weight_decay=0)
         for ep in range(0, args.epoch_per_stage):
            los_diff = self.trainEpoch_diffusion()
            log(self.makePrint('Train---[diffusion]', ep, args.epoch_per_stage, los_diff, 1))
            los_boos = self.trainEpoch_boost(stage, trained_model, opt)
            log(self.makePrint('Train--[boost]', ep, args.epoch_per_stage, los_boos, tstFlag))

         # phase 2
         model_tmp = Model(self.handler.image_feats.detach(), self.handler.text_feats.detach()).cuda()
         for i in self.model_name_list:
            model_tmp.model_dict[i].load_state_dict(self.model_ensemble.model_dict[i].state_dict())

         for ep in range(args.distill_epoch):
            los_dist = self.trainEpoch_distill(model_tmp, trained_model, trained_model_name, opt, stage)
            log(self.makePrint('Train---[distill]', ep, args.distill_epoch, los_dist, tstFlag))

         if tstFlag:
            reses = self.testEpoch_boost(stage)
            if (reses['Recall'] > recallMax):
               recallMax = reses['Recall']
               ndcgMax = reses['NDCG']
               precisionMax = reses['Precision']
               bestEpoch = stage + args.warmup_epoch
               torch.save(self.model_ensemble.model_dict['modal_consist'].state_dict(), os.path.join(save_path, "model_hm.pth"))
               torch.save(self.model_ensemble.model_dict['text_compl'].state_dict(), os.path.join(save_path, "model_tht.pth"))
               torch.save(self.model_ensemble.model_dict['visual_compl'].state_dict(), os.path.join(save_path, "model_vht.pth"))
               torch.save(self.diffusion_model.state_dict(), os.path.join(save_path, "diffusion_model.pth"))
               torch.save(self.denoise_model_image.state_dict(), os.path.join(save_path, "denoise_model_image.pth"))
               torch.save(self.denoise_model_text.state_dict(), os.path.join(save_path, "denoise_model_text.pth"))
            log(self.makePrint('Test', args.warmup_epoch + stage, args.warmup_epoch + args.stage, reses, tstFlag))

      print('Best epoch : ', bestEpoch, ' , Recall : ', recallMax, ' , NDCG : ', ndcgMax, ' , Precision', precisionMax)

   def prepareModel(self):
      self.model_ensemble = Model(self.handler.image_feats.detach(), self.handler.text_feats.detach()).cuda()

      self.opt = torch.optim.Adam(list(self.model_ensemble.model_dict['modal_consist'].parameters()) + list(
         self.model_ensemble.model_dict['visual_compl'].parameters()) + list(
         self.model_ensemble.model_dict['text_compl'].parameters()),
                                  lr=args.lr, weight_decay=0)

      self.diffusion_model = GaussianDiffusion(args.noise_scale, args.noise_min, args.noise_max, args.steps).cuda()

      out_dims = eval(args.dims) + [args.item]
      in_dims = out_dims[::-1]
      self.denoise_model_image = Denoise(in_dims, out_dims, args.d_emb_size, norm=args.norm).cuda()
      self.denoise_opt_image = torch.optim.Adam(self.denoise_model_image.parameters(), lr=args.lr, weight_decay=0)

      out_dims = eval(args.dims) + [args.item]
      in_dims = out_dims[::-1]
      self.denoise_model_text = Denoise(in_dims, out_dims, args.d_emb_size, norm=args.norm).cuda()
      self.denoise_opt_text = torch.optim.Adam(self.denoise_model_text.parameters(), lr=args.lr, weight_decay=0)

      self.criterion = nn.MSELoss()


      self.total_list = []
      self.modal_sh_re20_list = []
      self.visual_sp_re20_list = []
      self.text_sp_re20_list = []

   def normalizeAdj(self, mat):
      degree = np.array(mat.sum(axis=-1))
      dInvSqrt = np.reshape(np.power(degree, -0.5), [-1])
      dInvSqrt[np.isinf(dInvSqrt)] = 0.0
      dInvSqrtMat = sp.diags(dInvSqrt)
      return mat.dot(dInvSqrtMat).transpose().dot(dInvSqrtMat).tocoo()

   def buildUIMatrix(self, u_list, i_list, edge_list):
      mat = coo_matrix((edge_list, (u_list, i_list)), shape=(args.user, args.item), dtype=np.float32)

      a = sp.csr_matrix((args.user, args.user))
      b = sp.csr_matrix((args.item, args.item))

      mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
      mat = (mat != 0) * 1.0
      mat = (mat + sp.eye(mat.shape[0])) * 1.0
      mat = self.normalizeAdj(mat)

      idxs = torch.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
      vals = torch.from_numpy(mat.data.astype(np.float32))
      shape = torch.Size(mat.shape)

      return torch.sparse.FloatTensor(idxs, vals, shape).cuda()


   def trainEpoch_pretrain(self):
      self.trnLoader = self.handler.trnLoader
      self.trnLoader.dataset.negSampling()
      steps = self.trnLoader.dataset.__len__() // args.batch
      epLoss, epClLoss, epSPLoss, epSWDLoss = 0, 0, 0, 0

      for i, tem in enumerate(self.trnLoader):
         ancs, poss, negs = tem
         ancs = ancs.long().cuda()
         poss = poss.long().cuda()
         negs = negs.long().cuda()

         self.opt.zero_grad()
         '''
         cl loss
         '''
         item_embed_visual_sh, item_embed_visual_sp = self.model_ensemble.model_dict['modal_consist'].getImageFeats(), \
                                                      self.model_ensemble.model_dict['visual_compl'].getImageFeats()
         item_embed_text_sh, item_embed_text_sp = self.model_ensemble.model_dict['modal_consist'].getTextFeats(), \
                                                  self.model_ensemble.model_dict['text_compl'].getTextFeats()
         clLoss = (contrastLoss(item_embed_visual_sh, item_embed_text_sh, poss, args.temp)) * args.ssl_reg

         loss = clLoss
         epClLoss += clLoss.item()

         '''
         swd loss
         '''
         swdLoss = args.sw_reg * sliced_wasserstein_distance(item_embed_visual_sh[poss], item_embed_text_sh[poss],num_projections=args.n_projection)

         '''
         disparity loss
         '''
         # speLoss = 0
         dis_visual = self.criterion(item_embed_visual_sp[poss], item_embed_visual_sh[poss])
         dis_text = self.criterion(item_embed_text_sp[poss], item_embed_text_sh[poss])
         speLoss = -1 * torch.min(dis_visual, torch.tensor(args.maxd_v)) * args.sp_reg
         speLoss += -1 * torch.min(dis_text, torch.tensor(args.maxd_t)) * args.sp_reg

         loss += speLoss
         epSPLoss += speLoss.item()

         epSWDLoss += swdLoss

         loss += swdLoss
         epLoss += loss.item()

         loss.backward()
         self.opt.step()

         log('Step %d/%d: cl : %.3f ; dis_v : %.3f; dis_t : %.3f; sp : %.3f ; swd : %.3f \n' % (
            i,
            steps,
            clLoss.item(),
            (dis_visual),
            (dis_text),
            speLoss,
            swdLoss
         ), save=False, oneline=True)

         ret = dict()
         ret['Loss'] = epLoss / steps
         ret['CL loss'] = epClLoss / steps
         ret['SP loss'] = epSPLoss / steps
         ret['SWD loss'] = epSWDLoss / steps
      return ret

   def trainEpoch_boost(self, stage, trained_model, optimizer):
      opt = optimizer

      trained_model_name = self.model_name_list[stage % 3]
      if stage < 3:
         masked_model_name = self.model_name_list[stage:]
      else:
         masked_model_name = []

      self.trnLoader = self.handler.trnLoader
      self.trnLoader.dataset.negSampling()
      epLoss, epBoostLoss = 0, 0
      steps = self.trnLoader.dataset.__len__() // args.batch

      for i, tem in enumerate(self.trnLoader):
         ancs, poss, negs = tem
         ancs = ancs.long().cuda()
         poss = poss.long().cuda()
         negs = negs.long().cuda()

         opt.zero_grad()

         usrEmbeds_cur, itmEmbeds_cur = trained_model.forward_current(trained_model_name, self.handler.torchBiAdj,
                                                                      self.image_UI_matrix, self.text_UI_matrix)

         ancEmbeds_cur = usrEmbeds_cur[ancs]
         posEmbeds_cur = itmEmbeds_cur[poss]
         negEmbeds_cur = itmEmbeds_cur[negs]
         scoreDiff_cur = pairPredict(ancEmbeds_cur, posEmbeds_cur, negEmbeds_cur)
         if stage == 0:
            boostLoss = - (scoreDiff_cur).sigmoid().log().sum() / args.batch
         else:
            usrEmbeds_ensem, itmEmbeds_emsem = self.model_ensemble.forward_mask(masked_model_name,
                                                                                self.handler.torchBiAdj,
                                                                                self.image_UI_matrix,
                                                                                self.text_UI_matrix)
            usrEmbeds_ensem, itmEmbeds_emsem = usrEmbeds_ensem.detach(), itmEmbeds_emsem.detach()
            ancEmbeds_ensem = usrEmbeds_ensem[ancs]
            posEmbeds_ensem = itmEmbeds_emsem[poss]
            negEmbeds_ensem = itmEmbeds_emsem[negs]
            scoreDiff_ensem = pairPredict(ancEmbeds_ensem, posEmbeds_ensem, negEmbeds_ensem)
            boostLoss = self.criterion(scoreDiff_cur, args.step_size * (1 - scoreDiff_ensem.sigmoid())) #
         epBoostLoss += boostLoss.item()

         regLoss = (trained_model.model_dict[trained_model_name].reg_loss()) * args.reg
         loss = boostLoss + regLoss
         epLoss += loss.item()

         loss.backward()
         opt.step()

         log(
            'Step %d/%d: boost : %.3f' % (
               i,
               steps,
               boostLoss.item()
            ), save=False, oneline=True)

      ret = dict()
      ret['Loss'] = epLoss / steps
      ret['Boost Loss'] = epBoostLoss / steps
      return ret

   def trainEpoch_distill(self, model_tmp, trained_model, trained_model_name, opt, stage):
      epdistillLoss, epgcLoss = 0.0, 0.0
      steps = self.trnLoader.dataset.__len__() // args.batch
      if stage < 3:
         masked_model_name = self.model_name_list[stage + 1:]
      else:
         masked_model_name = []

      weight = nn.Parameter(torch.tensor(1.0, requires_grad=True))
      opt = torch.optim.Adam(list(self.model_ensemble.model_dict['modal_consist'].parameters()) + list(
         self.model_ensemble.model_dict['visual_compl'].parameters()) + list(
         self.model_ensemble.model_dict['text_compl'].parameters()) + [weight], lr=args.lr_distill, weight_decay=0)

      for i, tem in enumerate(self.trnLoader):
         ancs, poss, negs = tem
         ancs = ancs.long().cuda()
         poss = poss.long().cuda()
         negs = negs.long().cuda()

         opt.zero_grad()

         '''
         Distill Loss
         '''
         usrEmbeds_student, itmEmbeds_student = self.model_ensemble.forward_current(trained_model_name,
                                                                                    self.handler.torchBiAdj,
                                                                                    self.image_UI_matrix,
                                                                                    self.text_UI_matrix)
         ancEmbeds_student = usrEmbeds_student[ancs]
         posEmbeds_student = itmEmbeds_student[poss]
         negEmbeds_student = itmEmbeds_student[negs]
         scoreDiff_student = pairPredict(ancEmbeds_student, posEmbeds_student, negEmbeds_student)

         usrEmbeds_train, itmEmbeds_train = trained_model.forward_current(trained_model_name, self.handler.torchBiAdj,
                                                                          self.image_UI_matrix, self.text_UI_matrix)
         usrEmbeds_train, itmEmbeds_train = usrEmbeds_train.detach(), itmEmbeds_train.detach()
         ancEmbeds_train = usrEmbeds_train[ancs]
         posEmbeds_train = itmEmbeds_train[poss]
         negEmbeds_train = itmEmbeds_train[negs]
         scoreDiff_train = pairPredict(ancEmbeds_train, posEmbeds_train, negEmbeds_train)


         usrEmbeds_pre, itmEmbeds_pre = model_tmp.forward_current(trained_model_name, self.handler.torchBiAdj,
                                                                  self.image_UI_matrix, self.text_UI_matrix)
         usrEmbeds_pre, itmEmbeds_pre = usrEmbeds_pre.detach(), itmEmbeds_pre.detach()

         ancEmbeds_pre = usrEmbeds_pre[ancs]
         posEmbeds_pre = itmEmbeds_pre[poss]
         negEmbeds_pre = itmEmbeds_pre[negs]
         scoreDiff_pre = pairPredict(ancEmbeds_pre, posEmbeds_pre, negEmbeds_pre)
         distill_loss = self.criterion(scoreDiff_student, weight * scoreDiff_train + scoreDiff_pre) * args.distill_reg

         '''
         BPR Loss
         '''
         usrEmbeds_ensemble, itmEmbeds_ensemble = self.model_ensemble.forward_mask(masked_model_name,
                                                                                   self.handler.torchBiAdj,
                                                                                   self.image_UI_matrix,
                                                                                   self.text_UI_matrix)
         ancEmbeds_ensemble = usrEmbeds_ensemble[ancs]
         posEmbeds_ensemble = itmEmbeds_ensemble[poss]
         negEmbeds_ensemble = itmEmbeds_ensemble[negs]
         scoreDiff_ensemble = pairPredict(ancEmbeds_ensemble, posEmbeds_ensemble, negEmbeds_ensemble)
         gcLoss = -(scoreDiff_ensemble).sigmoid().log().sum() / args.batch

         regLoss = (self.model_ensemble.model_dict['modal_consist'].reg_loss() + self.model_ensemble.model_dict[
            'visual_compl'].reg_loss() + self.model_ensemble.model_dict['text_compl'].reg_loss()) * args.reg


         epdistillLoss += distill_loss.item()
         epgcLoss += gcLoss.item()

         loss = distill_loss + gcLoss + regLoss
         loss.backward()
         opt.step()

         log(
            'Step %d/%d: distill loss : %.3f ; gc loss : %.3f ' % (
               i,
               steps,
               distill_loss.item(),
               gcLoss.item()
            ), save=False, oneline=True)

      ret = dict()
      ret['Distill Loss'] = epdistillLoss / steps
      ret['GC Loss'] = epgcLoss / steps
      return ret

   def trainEpoch_diffusion(self):
      epDiLoss_image, epDiLoss_text, epDiLoss_audio = 0, 0, 0
      diffusionLoader = self.handler.diffusionLoader
      for i, batch in enumerate(diffusionLoader):
         batch_item, batch_index = batch
         batch_item, batch_index = batch_item.cuda(), batch_index.cuda()

         iEmbeds_v = self.model_ensemble.model_dict['modal_consist'].getItemEmbeds().detach()
         iEmbeds_t = self.model_ensemble.model_dict['modal_consist'].getItemEmbeds().detach()

         image_feats = self.model_ensemble.model_dict['modal_consist'].getImageFeats().detach()
         text_feats = self.model_ensemble.model_dict['modal_consist'].getTextFeats().detach()

         self.denoise_opt_image.zero_grad()
         self.denoise_opt_text.zero_grad()

         diff_loss_image, gc_loss_image = self.diffusion_model.training_losses(self.denoise_model_image, batch_item,
                                                                               iEmbeds_v, batch_index, image_feats)
         diff_loss_text, gc_loss_text = self.diffusion_model.training_losses(self.denoise_model_text, batch_item,
                                                                             iEmbeds_t, batch_index, text_feats)

         loss_image = diff_loss_image.mean() + gc_loss_image.mean() * args.e_loss
         loss_text = diff_loss_text.mean() + gc_loss_text.mean() * args.e_loss

         epDiLoss_image += loss_image.item()
         epDiLoss_text += loss_text.item()

         loss = loss_image + loss_text

         loss.backward()

         self.denoise_opt_image.step()
         self.denoise_opt_text.step()

         log('Diffusion Step %d/%d' % (i, diffusionLoader.dataset.__len__() // args.batch), save=False, oneline=True)
      ret = dict()
      ret['Di image loss'] = epDiLoss_image / (diffusionLoader.dataset.__len__() // args.batch)
      ret['Di text loss'] = epDiLoss_text / (diffusionLoader.dataset.__len__() // args.batch)

      with torch.no_grad():

         u_list_image = []
         i_list_image = []
         edge_list_image = []

         u_list_text = []
         i_list_text = []
         edge_list_text = []

         for _, batch in enumerate(diffusionLoader):
            batch_item, batch_index = batch
            batch_item, batch_index = batch_item.cuda(), batch_index.cuda()

            # image
            denoised_batch = self.diffusion_model.p_sample(self.denoise_model_image, batch_item, args.sampling_steps,
                                                           args.sampling_noise)
            top_item, indices_ = torch.topk(denoised_batch, k=args.rebuild_k)

            for i in range(batch_index.shape[0]):
               for j in range(indices_[i].shape[0]):
                  u_list_image.append(int(batch_index[i].cpu().numpy()))
                  i_list_image.append(int(indices_[i][j].cpu().numpy()))
                  edge_list_image.append(1.0)

            # text
            denoised_batch = self.diffusion_model.p_sample(self.denoise_model_text, batch_item, args.sampling_steps,
                                                           args.sampling_noise)
            top_item, indices_ = torch.topk(denoised_batch, k=args.rebuild_k)

            for i in range(batch_index.shape[0]):
               for j in range(indices_[i].shape[0]):
                  u_list_text.append(int(batch_index[i].cpu().numpy()))
                  i_list_text.append(int(indices_[i][j].cpu().numpy()))
                  edge_list_text.append(1.0)


         # image
         u_list_image = np.array(u_list_image)
         i_list_image = np.array(i_list_image)
         edge_list_image = np.array(edge_list_image)
         self.image_UI_matrix = self.buildUIMatrix(u_list_image, i_list_image, edge_list_image)
         self.image_UI_matrix = self.model_ensemble.model_dict['visual_compl'].edgeDropper(
            self.image_UI_matrix)  # 53955*53955
         # print(self.image_UI_matrix.shape)

         # text
         u_list_text = np.array(u_list_text)
         i_list_text = np.array(i_list_text)
         edge_list_text = np.array(edge_list_text)
         self.text_UI_matrix = self.buildUIMatrix(u_list_text, i_list_text, edge_list_text)
         self.text_UI_matrix = self.model_ensemble.model_dict['visual_compl'].edgeDropper(self.text_UI_matrix)


      return ret

   def testEpoch_boost(self, stage):
      tstLoader = self.handler.tstLoader
      epRecall, epNdcg, epPrecision = [0] * 3
      epRecall_sh, epRecall_vsp, epRecall_tsp = 0, 0, 0
      i = 0
      num = tstLoader.dataset.__len__()
      steps = num // args.tstBat


      usrEmbeds, itmEmbeds = self.model_ensemble.forward_mask([], self.handler.torchBiAdj, self.image_UI_matrix,
                                                              self.text_UI_matrix)
      usrEmbeds_sh, itmEmbeds_sh = self.model_ensemble.forward_current('modal_consist', self.handler.torchBiAdj,
                                                                         self.image_UI_matrix, self.text_UI_matrix)
      usrEmbeds_vsp, itmEmbeds_vsp = self.model_ensemble.forward_current('visual_compl', self.handler.torchBiAdj,
                                                                         self.image_UI_matrix, self.text_UI_matrix)
      usrEmbeds_tsp, itmEmbeds_tsp = self.model_ensemble.forward_current('text_compl', self.handler.torchBiAdj,
                                                                         self.image_UI_matrix, self.text_UI_matrix)

      for usr, trnMask in tstLoader:
         i += 1
         usr = usr.long().cuda()
         trnMask = trnMask.cuda()

         allPreds = torch.mm(usrEmbeds[usr], torch.transpose(itmEmbeds, 1, 0)) * (1 - trnMask) - trnMask * 1e8
         _, topLocs = torch.topk(allPreds, args.topk)
         recall, ndcg, precision = self.calcRes(topLocs.cpu().numpy(), self.handler.tstLoader.dataset.tstLocs, usr)
         epRecall += recall
         epNdcg += ndcg
         epPrecision += precision
         log('Steps %d/%d: recall = %.2f, ndcg = %.2f , precision = %.2f ' % (i, steps, recall, ndcg, precision),
             save=False, oneline=True)

         allPreds = torch.mm(usrEmbeds_sh[usr], torch.transpose(itmEmbeds_sh, 1, 0)) * (1 - trnMask) - trnMask * 1e8
         _, topLocs = torch.topk(allPreds, args.topk)
         recall_sh, ndcg, precision = self.calcRes(topLocs.cpu().numpy(), self.handler.tstLoader.dataset.tstLocs, usr)
         epRecall_sh += recall_sh
         log('Steps %d/%d (visual-shared): recall = %.2f' % (i, steps, recall_sh), save=False, oneline=True)

         allPreds = torch.mm(usrEmbeds_vsp[usr], torch.transpose(itmEmbeds_vsp, 1, 0)) * (1 - trnMask) - trnMask * 1e8
         _, topLocs = torch.topk(allPreds, args.topk)
         recall_vsp, ndcg, precision = self.calcRes(topLocs.cpu().numpy(), self.handler.tstLoader.dataset.tstLocs, usr)
         epRecall_vsp += recall_vsp
         log('Steps %d/%d (visual-specific): recall = %.2f' % (i, steps, recall_vsp), save=False, oneline=True)

         allPreds = torch.mm(usrEmbeds_tsp[usr], torch.transpose(itmEmbeds_tsp, 1, 0)) * (1 - trnMask) - trnMask * 1e8
         _, topLocs = torch.topk(allPreds, args.topk)
         recall_tsp, ndcg, precision = self.calcRes(topLocs.cpu().numpy(), self.handler.tstLoader.dataset.tstLocs, usr)
         epRecall_tsp += recall_tsp
         log('Steps %d/%d (text-specific): recall = %.2f' % (i, steps, recall_tsp), save=False, oneline=True)

      ret = dict()
      ret['Recall'] = epRecall / num
      ret['NDCG'] = epNdcg / num
      ret['Precision'] = epPrecision / num
      self.total_list.append(ret['Recall'])
      self.modal_sh_re20_list.append(epRecall_sh / num)
      self.visual_sp_re20_list.append(epRecall_vsp / num)
      self.text_sp_re20_list.append(epRecall_tsp / num)

      return ret

   def calcRes(self, topLocs, tstLocs, batIds):
      assert topLocs.shape[0] == len(batIds)
      allRecall = allNdcg = allPrecision = 0
      for i in range(len(batIds)):
         temTopLocs = list(topLocs[i])
         temTstLocs = tstLocs[batIds[i]]
         tstNum = len(temTstLocs)
         maxDcg = np.sum([np.reciprocal(np.log2(loc + 2)) for loc in range(min(tstNum, args.topk))])
         recall = dcg = precision = 0
         for val in temTstLocs:
            if val in temTopLocs:
               recall += 1
               dcg += np.reciprocal(np.log2(temTopLocs.index(val) + 2))
               precision += 1
         recall = recall / tstNum
         ndcg = dcg / maxDcg
         precision = precision / args.topk
         allRecall += recall
         allNdcg += ndcg
         allPrecision += precision
      return allRecall, allNdcg, allPrecision


def seed_it(seed):
   random.seed(seed)
   os.environ["PYTHONSEED"] = str(seed)
   np.random.seed(seed)
   torch.cuda.manual_seed(seed)
   torch.cuda.manual_seed_all(seed)
   torch.backends.cudnn.deterministic = True
   torch.backends.cudnn.benchmark = True
   torch.backends.cudnn.enabled = True
   torch.manual_seed(seed)


if __name__ == '__main__':
   seed_it(args.seed)

   save_path = args.checkpoints
   if not os.path.exists(save_path):
      os.makedirs(save_path)


   command = 'cp ' + 'Main.py ' + args.checkpoints
   os.system(command)
   command = 'cp ' + 'Model_d3er.py ' + args.checkpoints
   os.system(command)
   command = 'cp ' + 'Params.py ' + args.checkpoints
   os.system(command)

   os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
   logger.saveDefault = True

   log('Start')
   handler = DataHandler()
   handler.LoadData()
   log('Load Data')

   coach = Coach(handler)
   coach.run_boost()