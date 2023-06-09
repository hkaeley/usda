from dataset import Dataset
from argparse import ArgumentParser
import torch
from tqdm import tqdm
from model import Nutrition_MLP
import numpy as np
import sys, os
from pathlib import Path
import wandb 
from sklearn.metrics import r2_score
#TODO: Make dataloader for train, test so you can take advantage of batches later

class Trainer():
    def __init__(self, args):
        self.args = args
        self.epoch_idx = self.args.epochs
        if self.args.log_wandb == "True":
            wandb.init(project=self.args.wandb_project, entity=self.args.wandb_entity) 
    
    def build_dataset(self):
        if self.args.load_dataset == "True":
            self.dataset = Dataset(self.args.filename, self.args.nutrients_filepath, load_dataset = True)
            self.dataset = self.dataset.load(self.args.dataset_load_path)
        else:
            self.dataset = Dataset(self.args.filename, self.args.nutrients_filepath, load_dataset = False)
            self.dataset.save(self.args.dataset_save_path)
            

    def split_dataset(self):
        self.train_data_x, self.test_data_x, self.train_data_y, self.test_data_y = self.dataset.split_dataset()

    def build_model(self):
        if self.args.model == "nutrition_mlp":
            self.model = Nutrition_MLP(self.args.hidden_layers) #TODO: implement variable layers and layer dims
        else:
            raise ValueError("Model type not recognized")
        self.model = self.model.to(args.device)

    def train(self):
        if self.args.optimizer == "SGD":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=float(self.args.learning_rate)) 
        elif self.args.optimizer == "Adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(self.args.learning_rate)) 
        else:
            raise ValueError("Optimizer arg not recognized")
        
        if self.args.loss_func == "cross_entropy":
            self.loss_function = torch.nn.CrossEntropyLoss()
        elif self.args.loss_func == "mse": #want to use mse for regression
            self.loss_function = torch.nn.MSELoss()
        elif self.args.loss_func == "l1":
            self.loss_function = torch.nn.L1Loss()

        tqdm_bar = tqdm(range(self.args.epochs))

        if self.args.test == "True": #include last epoch as well
            torch.cuda.empty_cache()
            self.evaluate()   
            torch.cuda.empty_cache()


        for epoch_idx in tqdm_bar:
            self.epoch_idx = epoch_idx
            self.model.train() 
            for i in range(0, len(self.train_data_x), self.args.batch_size): # iterate through batches of the dataset
                self.optimizer.zero_grad() #reset opitmizer

                batch_index = i + self.args.batch_size if i + self.args.batch_size <= len(self.train_data_x) else len(self.train_data_x)
                input = torch.from_numpy(self.train_data_x[i:batch_index, :]).float().to(self.args.device) #load training batch
                ground_truth = torch.from_numpy(self.train_data_y[i:batch_index, :]).float().to(self.args.device)
    
                self.optimizer.zero_grad()

                if self.args.model in ["nutrition_mlp"]:
                    output = self.model(input)
                else:
                    raise ValueError('Model not recognized')
               
                loss = self.loss_function(output, ground_truth)

                loss.backward()
                self.optimizer.step()
                tqdm_bar.set_description('Epoch: {:d}, loss_train: {:.4f}'.format(self.epoch_idx, loss.detach().cpu().item()))
                torch.cuda.empty_cache()

            if self.epoch_idx % int(self.args.test_step) == 0 or self.epoch_idx == int(self.args.epochs) - 1: #include last epoch as well
                torch.cuda.empty_cache()
                self.evaluate()   
                torch.cuda.empty_cache()

        self.save_model(True) #save model once done training


    # def compute_roc_auc_score(self, y_true, y_pred):
    #     # if we take any two observations a and b such that a > b, then roc_auc_score is equal to the probability that our model actually ranks a higher than b

    #     num_same_sign = 0
    #     num_pairs = 0
        
    #     for a in range(len(y_true)):
    #         for b in range(len(y_true)):
    #             if y_true[a] > y_true[b]: #find pairs of data in which the true value of a is > true value of b
    #                 num_pairs += 1
    #                 if y_pred[a] > y_pred[b]: #if predicted value of a is greater then += 1 since we are correct
    #                     num_same_sign += 1
    #                 elif y_pred[a] == y_pred[b]: #case in which they are equal
    #                     num_same_sign += .5
                
    #     return num_same_sign / num_pairs

    # def compute_loss(self, y_true, y_pred):
    #     agg_loss = 0
    #     import pdb; pdb.set_trace()
    #     for gt, pred in zip(y_true, y_pred):

    #         #compute loss
    #         loss = self.loss_function(pred, gt)
    #         agg_loss += loss.detach().cpu().item()
    #     return agg_loss


    def compute_loss(self, y_true, y_pred):
        agg_loss = 0
        #assuming labels are formatted as [protein, fat, carb, energy] 
        protein_agg_loss = 0
        fat_agg_loss = 0
        carb_agg_loss = 0
        energy_agg_loss = 0
        for gt, pred in zip(y_true, y_pred):
            #compute loss
            loss = self.loss_function(pred, gt) #by default it is the average of all output dims
            agg_loss += loss.detach().cpu().item()

            loss = self.loss_function(pred[0], gt[0])
            protein_agg_loss += loss.detach().cpu().item()

            loss = self.loss_function(pred[1], gt[1])
            fat_agg_loss += loss.detach().cpu().item()

            loss = self.loss_function(pred[2], gt[2])
            carb_agg_loss += loss.detach().cpu().item()

            loss = self.loss_function(pred[3], gt[3])
            energy_agg_loss += loss.detach().cpu().item()
        return agg_loss, protein_agg_loss, fat_agg_loss, carb_agg_loss, energy_agg_loss

    # def compute_dir_acc(self, y_true, y_pred, x_data):
    #     correct = 0
    #     total = 0
    #     for gt, pred, data in zip(y_true, y_pred, x_data):
    #         total += 1
    #         if ((pred > data[-1]['open']) and (gt > data[-1]['open'])):
    #             correct += 1
    #         elif ((pred < data[-1]['open']) and (gt < data[-1]['open'])):
    #             correct += 1
    #         elif ((pred == data[-1]['open']) and (gt == data[-1]['open'])):
    #             correct += 1
    #     return correct/total

    def metrics(self, y_true, y_pred, x_data):
        #compute agg loss
        agg_loss, protein_agg_loss, fat_agg_loss, carb_agg_loss, energy_agg_loss = self.compute_loss(y_true, y_pred)

        #compute auc
        # auc = self.compute_roc_auc_score(y_true, y_pred)

        #compute r^2 accuracy

        # dir_acc = self.compute_dir_acc(y_true, y_pred, x_data)
        
        # return {'agg_loss': agg_loss, 'auc': auc, 'r2': r2_score(y_true, y_pred), 'dir_acc': dir_acc}
        y_true, y_pred = [batch.tolist() for batch in y_true], [batch.tolist() for batch in y_pred]
        return {'agg_loss': agg_loss, "protein_agg_loss": protein_agg_loss, "fat_agg_loss": fat_agg_loss, "carb_agg_loss": carb_agg_loss, "energy_agg_loss": energy_agg_loss,
                'r2': r2_score(y_true, y_pred)}


 
    def inference(self, x_data, y_data): #use dataloaders here instead once implemented
        agg_loss = 0
        y_pred = []
        y_true = []

        for i in range(0, len(x_data), self.args.batch_size): # iterate through batches of the dataset
            batch_index = i + self.args.batch_size if i + self.args.batch_size <= len(x_data) else len(x_data)
            input = torch.from_numpy(x_data[i:batch_index, :]).float().to(self.args.device) #load training batch
            ground_truth = torch.from_numpy(y_data[i:batch_index, :]).float().to(self.args.device)
            if self.args.model in ["nutrition_mlp"]:
                output = self.model(input)
            else:
                raise ValueError('Model not recognized')
            output = output.detach().cpu()
            ground_truth = ground_truth.detach().cpu()
            for batch in ground_truth:
                y_true.append(batch) #for sklearn r2 func
            for batch in output:
                y_pred.append(batch)
            torch.cuda.empty_cache()
        
        # import pdb; pdb.set_trace()
        return self.metrics(y_true, y_pred, x_data)

    '''runs inference on training and testing sets and collects scores''' #only log to wanb during eval since thats only when u get a validation loss
    def evaluate(self):
        self.model.eval()
        if self.args.epochs == 0: #if just doing prediction
            train_results = {}
            print('skipping training set.')
        else:
            train_results = self.inference(self.train_data_x, self.train_data_y)
            train_results.update({'train_avg_loss': train_results["agg_loss"]/len(self.train_data_y)})
            train_results.update({'train_protein_avg_loss': train_results["protein_agg_loss"]/len(self.train_data_y)})
            train_results.update({'train_fat_avg_loss': train_results["fat_agg_loss"]/len(self.train_data_y)})
            train_results.update({'train_carb_avg_loss': train_results["carb_agg_loss"]/len(self.train_data_y)})
            train_results.update({'train_energy_avg_loss': train_results["energy_agg_loss"]/len(self.train_data_y)})
            # train_results.update({'train_auc': train_results["auc"]})
            train_results.update({'train_r2': train_results["r2"]})
            # train_results.update({'train_dir_acc': train_results["dir_acc"]})
            print("train_avg_loss: " + str(train_results['train_avg_loss']))
            print("train_protein_avg_loss: " + str(train_results['train_protein_avg_loss']))
            print("train_fat_avg_loss " + str(train_results['train_fat_avg_loss']))
            print("train_carb_avg_loss: " + str(train_results['train_carb_avg_loss']))
            print("train_energy_avg_loss: " + str(train_results['train_energy_avg_loss']))
            # print("train auc: " + str(train_results['auc']))
            print("train r2: " + str(train_results['r2']))
            # print("train dir_acc: " + str(train_results['dir_acc']))

        val_results = self.inference(self.test_data_x, self.test_data_y)
        val_results.update({'test_avg_loss': val_results["agg_loss"]/len(self.test_data_y)})
        val_results.update({'test_protein_avg_loss': val_results["protein_agg_loss"]/len(self.test_data_y)})
        val_results.update({'test_fat_avg_loss': val_results["fat_agg_loss"]/len(self.test_data_y)})
        val_results.update({'test_carb_avg_loss': val_results["carb_agg_loss"]/len(self.test_data_y)})
        val_results.update({'test_energy_avg_loss': val_results["energy_agg_loss"]/len(self.test_data_y)})
        # val_results.update({'val_auc': val_results["auc"]})
        val_results.update({'val_r2': val_results["r2"]})
        # val_results.update({'val_dir_acc': val_results["dir_acc"]})
        print("test_avg_loss: " + str(val_results['test_avg_loss']))
        print("test_protein_avg_loss: " + str(val_results['test_protein_avg_loss']))
        print("test_fat_avg_loss: " + str(val_results['test_fat_avg_loss']))
        print("test_carb_avg_loss: " + str(val_results['test_carb_avg_loss']))
        print("test_energy_avg_loss: " + str(val_results['test_energy_avg_loss']))
        # print("val auc: " + str(val_results['auc']))
        print("val r2: " + str(val_results['r2']))
        # print("val dir_acc: " + str(val_results['dir_acc']))

        #train_results.update({'epoch': self.epoch_idx})
        val_results.update({'epoch': self.epoch_idx})

        #combine train and val results into one to make logging easier
        #   only log both during inference
        val_results.update(train_results)
        if self.args.log_wandb == "True":
            wandb.log(val_results)
        else:
            print(val_results)


    def save_model(self, is_best=False):
        if is_best:
            saved_path = Path(self.args.model_save_directory + self.args.model_save_file).resolve()
        else:
            saved_path = Path(self.args.model_save_directory + self.args.model_save_file).resolve()
        os.makedirs(os.path.dirname(saved_path), exist_ok=True)
        torch.save({
            'epoch': self.epoch_idx,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            #'loss': self.metrics.best_val_loss,
        }, str(saved_path))
        with open(os.path.dirname(saved_path) + "/model_parameters.txt", "w+") as f:
            f.write(str(self.args.__dict__))
            f.write('\n')
            f.write(str(' '.join(sys.argv)))
        print("Model saved.")


    '''Function to load the model, optimizer, scheduler.'''
    def load_model(self):  
        saved_path = Path(self.args.model_load_path).resolve()
        if saved_path.exists():
            self.build_model()
            torch.cuda.empty_cache()
            checkpoint = torch.load(str(saved_path), map_location="cpu")
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epoch_idx = checkpoint['epoch']
            #self.metrics.best_val_loss = checkpoint['loss']
            self.model.to(self.args.device)
            self.model.eval()
        else:
            raise FileNotFoundError("model load path does not exist.")
        print("Model loaded from file.")



if __name__ == "__main__":
        ap = ArgumentParser(description='The parameters for training.') 
        ap.add_argument('--filename', type=str, default="/content/dsusda/branded_food.csv", help="Load dataset pkl.")
        ap.add_argument('--nutrients_filepath', type=str, default="/content/dsusda/food_nutrient.csv", help="Load dataset pkl.")
        ap.add_argument('--load_dataset', type=str, default="False", help="Load dataset pkl.")
        ap.add_argument('--dataset_load_path', type=str, default="usda_ds.pkl", help="The path defining location to load dataset object from.")
        ap.add_argument('--dataset_save_path', type=str, default="usda_ds.pkl", help="The path defining location to save dataset object to.")

        ap.add_argument('--hidden_layers', type=str, default = "512, 256, 128, 64, 32, 16", help='Hidden Layer Dims')

        ap.add_argument('--epochs', type=int, default = 25)
        ap.add_argument('--device', type=str, default = "cuda:0")
        ap.add_argument('--test_step', type=int, default = 5)
        ap.add_argument('--batch_size', type=int, default = 8)
        ap.add_argument('--optimizer', type=str, default = "Adam")
        ap.add_argument('--loss_func', type=str, default = "mse")
        ap.add_argument('--learning_rate', type=float, default = 0.0001)
        ap.add_argument('--log_wandb', type=str, default = "True")

        ap.add_argument('--test', type=str, default = "Test Evaluation Before Training")
        ap.add_argument('--wandb_project', type=str, default = "test-project")
        ap.add_argument('--wandb_entity', type=str, default = "h199_research")
        ap.add_argument('--model_save_directory', type=str, default = "saved_models/")
        ap.add_argument('--model_save_file', type=str, default = "model_best_val_loss_.vec.pt")
        ap.add_argument('--model_load_path', type=str, default = "saved_models/model_best_val_loss_.vec.pt")
        ap.add_argument('--load_model', type=bool, default = False)
        ap.add_argument('--model', type=str, default = "nutrition_mlp")
        #include more args here if we decide to do ensemble training?


        args = ap.parse_args()
        args.hidden_layers = [int(item) for item in args.hidden_layers.split(',')]
        trainer = Trainer(args)
        trainer.build_dataset()
        trainer.split_dataset()
        trainer.build_model()
        trainer.train()
