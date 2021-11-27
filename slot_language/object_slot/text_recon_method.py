from obj_method import ObjSlotAttentionVideoLanguageMethod


class ObjTextReconSlotAttentionVideoLanguageMethod(
        ObjSlotAttentionVideoLanguageMethod):

    def __init__(self, model, datamodule, params):
        super().__init__(model, datamodule, params)
