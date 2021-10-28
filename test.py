from rederer_tmp.contrastive_unpaired_translation.options.train_options import TrainOptions
from rederer_tmp.contrastive_unpaired_translation.data import create_dataset


if __name__ == '__main__':
    opt = TrainOptions().parse()
    dataset = create_dataset(opt)
    dataset_size = len(dataset)

    print('The number of training images = %d' % dataset_size)

    for i, data in enumerate(dataset):
        a_data = data["A"]
        b_data = data["B"]
        print(a_data.size())
        print(b_data.size())
        break
