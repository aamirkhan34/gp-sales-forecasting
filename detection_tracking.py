
class DetectionTracking(object):
    label_wise_tracker = {}

    def __init__(self, labels):
        for label in labels:
            self.label_wise_tracker[str(label)] = {}

    def set_label_wise_tracker_tp(self, label, gen, classified):
        label = str(label)
        gen = str(gen)

        if classified:
            if self.label_wise_tracker[label].get(gen) and self.label_wise_tracker[label][gen].get("tp"):
                self.label_wise_tracker[label][gen]["tp"] = self.label_wise_tracker[label][gen]["tp"] + 1
            else:
                self.label_wise_tracker[label][gen]["tp"] = 1

    def set_label_wise_tracker_total(self, label, gen):
        label = str(label)
        gen = str(gen)

        if self.label_wise_tracker[label].get(gen) and self.label_wise_tracker[label][gen].get("total"):
            self.label_wise_tracker[label][gen]["total"] = self.label_wise_tracker[label][gen]["total"] + 1
        else:
            self.label_wise_tracker[label][gen] = {}
            self.label_wise_tracker[label][gen]["total"] = 1

    def get_label_wise_tracker(self):
        return self.label_wise_tracker
