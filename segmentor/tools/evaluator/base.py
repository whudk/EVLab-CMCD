from lib.utils.tools.logger import Logger as Log


class _BaseEvaluator:

    def __init__(self, configer, trainer, num_classes=None):
        self.configer = configer
        self.trainer = trainer
        self.num_classes = (
            num_classes
            if num_classes is not None
            else self.configer.get("data", "num_classes")
        )
        self._init_running_scores()
        self.conditions = configer.conditions

    def print_scores(self, show_miou=True):
        rs = self.running_scores["seg"]
        Log.info(f"Mean IOU: {rs.get_mean_iou()}")
        Log.info(f"Pixel ACC: {rs.get_pixel_acc()}")
        Log.info(f"Class IOU: {rs.get_cls_iou()}")
        Log.info("F1 Score: {} Precision: {} Recall: {}".format(*rs.get_F1_score()))

    def update_acc(self):
        try:
            rs = self.running_scores[self.save_net_main_key]
            acc = rs.get_pixel_acc()
            max_acc = self.configer.get("max_accuracy")
            self.configer.update(["accuracy"], acc)
            if acc > max_acc:
                Log.info("acc {} -> {}".format(max_acc, acc))
                self.configer.update(["max_accuracy"], acc)
        except Exception as e:
            Log.warn(e)

    def update_performance(self):

        try:
            rs = self.running_scores[self.save_net_main_key]
            if self.save_net_metric == "miou":
                perf = rs.get_mean_iou()
            elif self.save_net_metric == "acc":
                perf = rs.get_pixel_acc()
            max_perf = self.configer.get("max_performance")
            self.configer.update(["performance"], perf)
            if perf > max_perf:
                Log.info("Performance {} -> {}".format(max_perf, perf))
                self.configer.update(["max_performance"], perf)
        except Exception as e:
            Log.warn(e)

    def reset(self):
        for rs in self.running_scores.values():
            rs.reset()
