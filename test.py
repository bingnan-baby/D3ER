from DataHandler import *
from Model_d3er import *
from Params import args
import torch, os
import pandas as pd


def normalizeAdj(mat):
   degree = np.array(mat.sum(axis=-1))
   dInvSqrt = np.reshape(np.power(degree, -0.5), [-1])
   dInvSqrt[np.isinf(dInvSqrt)] = 0.0
   dInvSqrtMat = sp.diags(dInvSqrt)
   return mat.dot(dInvSqrtMat).transpose().dot(dInvSqrtMat).tocoo()

def buildUIMatrix(u_list, i_list, edge_list):
   mat = coo_matrix((edge_list, (u_list, i_list)), shape=(args.user, args.item), dtype=np.float32)

   a = sp.csr_matrix((args.user, args.user))
   b = sp.csr_matrix((args.item, args.item))

   mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
   mat = (mat != 0) * 1.0
   mat = (mat + sp.eye(mat.shape[0])) * 1.0
   mat = normalizeAdj(mat)

   idxs = torch.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
   vals = torch.from_numpy(mat.data.astype(np.float32))
   shape = torch.Size(mat.shape)

   return torch.sparse.FloatTensor(idxs, vals, shape).cuda()


def calcRes(topLocs, tstLocs, batIds):
   assert topLocs.shape[0] == len(batIds)
   allRecall = allNdcg = allPrecision = 0
   num_gt = 0
   for i in range(len(batIds)):
      temTopLocs = list(topLocs[i])
      temTstLocs = tstLocs[batIds[i]]
      tstNum = len(temTstLocs)
      num_gt += tstNum
      # print(tstNum)
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
   return allRecall, allNdcg, allPrecision, num_gt

handler = DataHandler()
handler.LoadData()

diffusionLoader = handler.diffusionLoader

model = Model(handler.image_feats.detach(), handler.text_feats.detach()).cuda()
diffusion_model = GaussianDiffusion(args.noise_scale, args.noise_min, args.noise_max, args.steps).cuda()
out_dims = eval(args.dims) + [args.item]
in_dims = out_dims[::-1]
denoise_model_image = Denoise(in_dims, out_dims, args.d_emb_size, norm=args.norm).cuda()
denoise_model_text = Denoise(in_dims, out_dims, args.d_emb_size, norm=args.norm).cuda()

ckpt_pth = "./checkpoints/baby/"
model.load_state_dict(torch.load(os.path.join(ckpt_pth,"model.pth")))
denoise_model_image.load_state_dict(torch.load(os.path.join(ckpt_pth,"denoise_model_image.pth")))
denoise_model_text.load_state_dict(torch.load(os.path.join(ckpt_pth,"denoise_model_text.pth")))
model.eval()

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
      denoised_batch = diffusion_model.p_sample(denoise_model_image, batch_item, args.sampling_steps,
                                                     args.sampling_noise)
      top_item, indices_ = torch.topk(denoised_batch, k=args.rebuild_k)

      for i in range(batch_index.shape[0]):
         for j in range(indices_[i].shape[0]):
            u_list_image.append(int(batch_index[i].cpu().numpy()))
            i_list_image.append(int(indices_[i][j].cpu().numpy()))
            edge_list_image.append(1.0)

      # text
      denoised_batch = diffusion_model.p_sample(denoise_model_text, batch_item, args.sampling_steps,
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
   image_UI_matrix = buildUIMatrix(u_list_image, i_list_image, edge_list_image)
   image_UI_matrix = model.edgeDropper(image_UI_matrix)

   # text
   u_list_text = np.array(u_list_text)
   i_list_text = np.array(i_list_text)
   edge_list_text = np.array(edge_list_text)
   text_UI_matrix = buildUIMatrix(u_list_text, i_list_text, edge_list_text)
   text_UI_matrix = model.edgeDropper(text_UI_matrix)

tstLoader = handler.tstLoader
epRecall, epNdcg, epPrecision = [0] * 3
i = 0
num = tstLoader.dataset.__len__()
steps = num // args.tstBat


results = []

usrEmbeds, itmEmbeds,usrEmbeds_sh, itmEmbeds_sh, usrEmbeds_vsp, itmEmbeds_vsp, usrEmbeds_tsp, itmEmbeds_tsp = model.forward_MM(handler.torchBiAdj, image_UI_matrix, text_UI_matrix)
for usr, trnMask in tstLoader:
   i += 1
   usr = usr.long().cuda()
   trnMask = trnMask.cuda()

   allPreds = torch.mm(usrEmbeds[usr], torch.transpose(itmEmbeds, 1, 0)) * (1 - trnMask) - trnMask * 1e8
   _, topLocs = torch.topk(allPreds, args.topk)
   recall, ndcg, precision, tstNum = calcRes(topLocs.cpu().numpy(), handler.tstLoader.dataset.tstLocs, usr)
   epRecall += recall
   epNdcg += ndcg
   epPrecision += precision
   print('Steps %d/%d: recall = %.2f, ndcg = %.2f , precision = %.2f , ItemNum = %.d ' % (i, steps, recall, ndcg, precision, tstNum))



ret = dict()
ret['Recall'] = epRecall / num
ret['NDCG'] = epNdcg / num
ret['Precision'] = epPrecision / num

print(' , Recall : ', ret['Recall'], ' , NDCG : ', ret['NDCG'], ' , Precision', ret['Precision'])