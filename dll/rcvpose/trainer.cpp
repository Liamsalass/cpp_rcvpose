#include "trainer.h"

using namespace std;

Trainer::Trainer(Options options)
{
	opts = options;


    // Instantiate the model

}

void Trainer::train()
{
    cout << "Starting Training Cycle" << endl;
    cout << "Setting up dataset loader" << endl;
    // Instantiate the dataset
    try {
        auto train_dataset = RData(opts.root_dataset, opts.dname, "train", opts.class_name, opts.kpt_num);


        // Instantiate the dataloaders 
        auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
            std::move(train_dataset),
            torch::data::DataLoaderOptions().batch_size(opts.batch_size).workers(1)
        );
    }
    catch (const torch::Error& e) {
        cout << "Error: " << e.msg() << endl;
        return;
    }

    // Instantiate the model
}

void Trainer::test()
{
	cout << "Starting Testing Cycle" << endl;
    cout << "Setting up dataset loader" << endl;
    //Set up dataloaders
    try {
        auto val_dataset = RData(opts.root_dataset, opts.dname, "test", opts.class_name, opts.kpt_num);
        auto val_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
            std::move(val_dataset),
            torch::data::DataLoaderOptions().batch_size(opts.batch_size).workers(1)
        );
    }
    catch (const torch::Error& e) {
        cout << "Error: " << e.msg() << endl;
        return;
    }
}
