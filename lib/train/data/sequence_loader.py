import torch.utils.data.dataloader

if float(torch.__version__[:3]) >= 1.9 or len('.'.join((torch.__version__).split('.')[0:2])) > 3:
    int_classes = int
else:
    from torch._six import int_classes


def slt_collate(batch):
    ret = {}
    for k in batch[0].keys():
        here_list = []
        for ex in batch:
            here_list.append(ex[k])
        ret[k] = here_list
    return ret


class SLTLoader(torch.utils.data.dataloader.DataLoader):
    """
    Data loader. Combines a dataset and a sampler, and provides
    single- or multi-process iterators over the dataset.
    """

    __initialized = False

    def __init__(self, name, dataset, training=True, batch_size=1, shuffle=False, sampler=None, batch_sampler=None,
                 num_workers=0, epoch_interval=1, collate_fn=None, stack_dim=0, pin_memory=False, drop_last=False,
                 timeout=0, worker_init_fn=None):
        if collate_fn is None:
            collate_fn = slt_collate

        super(SLTLoader, self).__init__(dataset, batch_size, shuffle, sampler, batch_sampler,
                                        num_workers, collate_fn, pin_memory, drop_last,
                                        timeout, worker_init_fn)

        self.name = name
        self.training = training
        self.epoch_interval = epoch_interval
        self.stack_dim = stack_dim
