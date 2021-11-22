from obj_method import ObjSlotAttentionVideoLanguageMethod


class ObjTextReconSlotAttentionVideoLanguageMethod(
        ObjSlotAttentionVideoLanguageMethod):

    def __init__(self, model, datamodule, params):
        super().__init__(model, datamodule, params)
        self.recon_feats = params.recon_feats
        if self.recon_feats:
            self.text_recon_loss = params.text_recon_loss
        else:
            self.color_cls_loss = params.color_cls_loss
            self.shape_cls_loss = params.shape_cls_loss

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        train_loss = self.model.loss_function(batch)
        if self.recon_feats:
            loss = train_loss['recon_loss'] + \
                self.text_recon_loss * train_loss['text_recon_loss']
        else:
            loss = train_loss['recon_loss'] + \
                self.color_cls_loss * train_loss['color_cls_loss'] + \
                self.shape_cls_loss * train_loss['shape_cls_loss']
        if 'entropy' in train_loss.keys():
            loss = loss + train_loss['entropy'] * self.entropy_loss_w
        train_loss['loss'] = loss
        logs = {key: val.item() for key, val in train_loss.items()}
        # record training time
        logs['data_time'] = \
            self.trainer.profiler.recorded_durations['get_train_batch'][-1]
        self.log_dict(logs, sync_dist=True)
        return {'loss': loss}
