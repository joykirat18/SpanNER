# encoding: utf-8


import os
from pytorch_lightning import Trainer

# from trainer_spanPred import BertLabeling # old evaluation version
# from trainer_spanPred_newEval import BertLabeling # new evaluation version
from trainer import BertNerTagger # start 0111

def evaluate(ckpt, hparams_file):
	"""main"""

	trainer = Trainer(gpus=[2])
	# trainer = Trainer(distributed_backend="dp")

	model = BertNerTagger.load_from_checkpoint(
		checkpoint_path=ckpt,
		hparams_file=hparams_file,
		map_location=None,
		batch_size=1,
		max_length=128,
		workers=32,
        bert_config_dir='bert-large-uncased',
        data_dir='data/fepd-complete',
        bert_dropout=0.2,
        model_dropout=0.2,
        span_combination_mode='x,y',
        max_spanLen=15,
        n_class=10,
        tokenLen_emb_dim=200,
        spanLen_emb_dim=300,
        morph_emb_dim=300,
        use_prune=False,
        use_spanLen=True,
        use_morph=False,
        use_span_weight=False,
        neg_span_weight=0.5,
        morph2idx_list=[('isupper', 1), ('islower', 2), ('istitle', 3), ('isdigit', 4), ('other', 5)],
        optimizer='adamw',
        fp_epoch_result = 'epoch_results.txt',
        bert_max_length=512,
        dataname='fepd',
        label2idx_list=[('O', 0), ('target_direct', 1), ('source_direct', 2), ('source_indirect', 3), ('data', 4), ('reason', 5), ('data_compulsory', 6), ('medium', 7), ('target_in_direct', 8), ('data_optional', 9)]
	)
	trainer.test(model=model)


if __name__ == '__main__':

	# root_dir1 = "/home/jlfu/SPred/train_logs"

	# datas = ['notenw','notewb','notebc','notemz','notetc','notebn','conll02dutch'] #
	# # datas = ['conll02spanish']  #
	# datas = ['wnut17']  #
	# for data in datas:
	# 	root_dir = os.path.join(root_dir1, data)
	# 	files = os.listdir(root_dir)
	# 	for file in files:
	# 		print('file:',file)
	# 		if "spanPred_bert" in file:
	# 			fmodel = os.path.join(root_dir,file)
	# 			fnames = os.listdir(fmodel)
	# 			ckmodel = ''
	# 			for fname in fnames:
	# 				if '.ckpt' in fname:
	# 					ckmodel= fname
	# 			CHECKPOINTS= os.path.join(fmodel,ckmodel)
	# 			HPARAMS = fmodel+"/lightning_logs/version_0/hparams.yaml"
	# 			evaluate(ckpt=CHECKPOINTS, hparams_file=HPARAMS)





	# # 0125
	# # conll03 bert-large, 9245 evaluation
	# midpath = "conll03/spanPred_bert-large-uncased_prunTrue_spLenTrue_spMorphFalse_SpWtFalse_value1_25149666_9245"
	# model_names = ["epoch=18.ckpt"]
	# for mn in model_names:
	# 	print("model-name: ",mn)
	# 	CHECKPOINTS = "/home/jlfu/SPred/train_logs/" + midpath + "/" +mn
	# 	HPARAMS = "/home/jlfu/SPred/train_logs/" + midpath + "/lightning_logs/version_0/hparams.yaml"
	# 	evaluate(ckpt=CHECKPOINTS, hparams_file=HPARAMS)

	# # 0125
	# # conll03 bert-large, 9245 evaluation
	# midpath = "wnut16/spanPred_bert-large-uncased_prunTrue_spLenFalse_spMorphFalse_SpWtFalse_value1_12635765"
	# model_names = ["epoch=11.ckpt"]
	# for mn in model_names:
	# 	print("model-name: ",mn)
	# 	CHECKPOINTS = "/home/jlfu/SPred/train_logs/" + midpath + "/" +mn
	# 	HPARAMS = "/home/jlfu/SPred/train_logs/" + midpath + "/lightning_logs/version_0/hparams.yaml"
	# 	evaluate(ckpt=CHECKPOINTS, hparams_file=HPARAMS)

	# 0129
	# # conll03 bert-large, 9252 evaluation
	# midpath = "conll03/spanPred_dev2train_bert-large-uncased_maxSpan4prunFalse_spLenTrue_spMorphFalse_SpWtFalse_value1_35462812"
	# model_names = ["epoch=18.ckpt"]
	# for mn in model_names:
	# 	print("model-name: ",mn)
	# 	CHECKPOINTS = "/home/jlfu/SPred/train_logs/" + midpath + "/" +mn
	# 	HPARAMS = "/home/jlfu/SPred/train_logs/" + midpath + "/lightning_logs/version_0/hparams.yaml"
	# 	evaluate(ckpt=CHECKPOINTS, hparams_file=HPARAMS)

	# # 0129
	# # conll03 bert-large, base 9157, prune evaluation
	# midpath = "conll03/spanPred_bert-large-uncased_maxSpan4prunFalse_spLenFalse_spMorphFalse_SpWtFalse_value1_61661034"
	# model_names = ["epoch=13.ckpt"]
	# for mn in model_names:
	# 	print("model-name: ",mn)
	# 	CHECKPOINTS = "/home/jlfu/SPred/train_logs/" + midpath + "/" +mn
	# 	HPARAMS = "/home/jlfu/SPred/train_logs/" + midpath + "/lightning_logs/version_0/hparams_prune.yaml"
	# 	evaluate(ckpt=CHECKPOINTS, hparams_file=HPARAMS)

	# # conll03 bert-large, base+len 9222, prune evaluation
	# midpath = "conll03/spanPred_bert-large-uncased_maxSpan4prunFalse_spLenTrue_spMorphFalse_SpWtFalse_value1_09854370"
	# model_names = ["epoch=13.ckpt"]
	# for mn in model_names:
	# 	print("model-name: ", mn)
	# 	CHECKPOINTS = "/home/jlfu/SPred/train_logs/" + midpath + "/" + mn
	# 	HPARAMS = "/home/jlfu/SPred/train_logs/" + midpath + "/lightning_logs/version_0/hparams_prune.yaml"
	# 	evaluate(ckpt=CHECKPOINTS, hparams_file=HPARAMS)

	# # # 0130
	# # wnut17 bert-large, base 9157, prune evaluation
	# midpath = "wnut17/spanPred_bert-large-uncased_prunFalse_spLenTrue_spMorphFalse_SpWtFalse_value1_96521534"
	# model_names = ["epoch=26.ckpt"]
	# for mn in model_names:
	# 	print("model-name: ", mn)
	# 	CHECKPOINTS = "/home/jlfu/SPred/train_logs/" + midpath + "/" + mn
	# 	HPARAMS = "/home/jlfu/SPred/train_logs/" + midpath + "/lightning_logs/version_0/hparams_prune.yaml"
	# 	evaluate(ckpt=CHECKPOINTS, hparams_file=HPARAMS)
	#
	# # # 0130
	# # wnut17 bert-large, base 9157, prune evaluation
	# midpath = "wnut17/spanPred_bert-large-uncased_prunFalse_spLenFalse_spMorphFalse_SpWtFalse_value1_09063161"
	# model_names = ["epoch=13.ckpt"]
	# for mn in model_names:
	# 	print("model-name: ", mn)
	# 	CHECKPOINTS = "/home/jlfu/SPred/train_logs/" + midpath + "/" + mn
	# 	HPARAMS = "/home/jlfu/SPred/train_logs/" + midpath + "/lightning_logs/version_0/hparams_prune.yaml"
	# 	evaluate(ckpt=CHECKPOINTS, hparams_file=HPARAMS)

	# 0130
	# # conll03 bert-large, base 9321, prune evaluation
	# midpath = "conll03/spanPred_dev2train_bert-large-uncased_prunTrue_spLenFalse_spMorphFalse_SpWtFalse_value1_35932770_9321"
	# model_names = ["epoch=5.ckpt"]
	# for mn in model_names:
	# 	print("model-name: ", mn)
	# 	CHECKPOINTS = "/home/jlfu/SPred/train_logs/" + midpath + "/" + mn
	# 	HPARAMS = "/home/jlfu/SPred/train_logs/" + midpath + "/lightning_logs/version_0/hparams_prune.yaml"
	# 	evaluate(ckpt=CHECKPOINTS, hparams_file=HPARAMS)

# # conll03 bert-large, base 9321, prune evaluation
# 	midpath = "conll03/spanPred_dev2train_bert-large-uncased_prunTrue_spLenTrue_spMorphTrue_SpWtFalse_value1_76851666_9318"
# 	model_names = ["epoch=14.ckpt"]
# 	for mn in model_names:
# 		print("model-name: ", mn)
# 		CHECKPOINTS = "/home/jlfu/SPred/train_logs/" + midpath + "/" + mn
# 		HPARAMS = "/home/jlfu/SPred/train_logs/" + midpath + "/lightning_logs/version_0/hparams_prune.yaml"
# 		evaluate(ckpt=CHECKPOINTS, hparams_file=HPARAMS)

#

	# # conll02spanish bert-large, base 0.873509, prune evaluation
	# midpath = "conll02spanish/spanPred_bert-base-multilingual-uncased_maxSpan4prunFalse_spLenFalse_spMorphFalse_SpWtFalse_value1_62640970"
	# model_names = ["epoch=5_v0.ckpt"]
	# for mn in model_names:
	# 	print("model-name: ", mn)
	# 	CHECKPOINTS = "/home/jlfu/SPred/train_logs/" + midpath + "/" + mn
	# 	HPARAMS = "/home/jlfu/SPred/train_logs/" + midpath + "/lightning_logs/version_0/hparams.yaml"
	# 	evaluate(ckpt=CHECKPOINTS, hparams_file=HPARAMS)

	# 0125
	# conll03 bert-large, 9245 evaluation
	midpath = "fepd/spanner_bert-large-uncased_spMLen_usePruneFalse_useSpLenTrue_useSpMorphFalse_SpWtFalse_value0.5_70270825"
	model_names = ["epoch=39-step=13518.ckpt"]
	for mn in model_names:
		print("model-name: ", mn)
		CHECKPOINTS = "train_logs/" + midpath + "/lightning_logs/version_0/checkpoints/" + mn
		HPARAMS = "train_logs/" + midpath + "/lightning_logs/version_0/hparams_prune.yaml"
		evaluate(ckpt=CHECKPOINTS, hparams_file=HPARAMS)



