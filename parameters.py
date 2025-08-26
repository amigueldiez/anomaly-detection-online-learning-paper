class Parameters:
    def __init__(self, features_dataset,scaling_type, n_flows_train_scaler,nu_value,q_value,learning_rate,flows_train_oml,accuracy,recall,false_positive):
        self.features_dataset = features_dataset  
        self.scaling_type = scaling_type
        self.n_flows_train_scaler = n_flows_train_scaler
        self.nu_value = nu_value
        self.q_value = q_value
        self.learning_rate = learning_rate
        self.flows_train_oml = flows_train_oml
        self.accuracy = accuracy
        self.recall = recall
        self.false_positive = false_positive
    



    def get_features_dataset(self):
        return self.features_dataset
    
    def set_features_dataset(self, new_features_dataset):
        self.features_dataset = new_features_dataset 



    def get_scaling_type(self):
        return self.scaling_type
    
    def set_scaling_type(self, new_scaling_type):
        self.scaling_type = new_scaling_type

    
    def get_n_flows_train_scaler(self):
        return self.n_flows_train_scaler
    
    def set_n_flows_train_scaler(self, new_n_flows_train_scaler):
        self.n_flows_train_scaler = new_n_flows_train_scaler
    
    def get_nu_value(self):
        return self.nu_value
    
    def set_nu_value(self, new_nu_value):
        self.nu_value = new_nu_value

    def get_q_value(self):
        return self.q_value
    
    def set_q_value(self, new_q_value):
        self.q_value = new_q_value

    def get_learning_rate(self):
        return self.learning_rate
    
    def set_learning_rate(self, new_learning_rate):
        self.learning_rate = new_learning_rate
    
    def get_flows_train_oml(self):
        return self.flows_train_oml
    
    def set_flows_train_oml(self, new_flows_train_oml):
        self.flows_train_oml = new_flows_train_oml
    
    

    
    def get_accuracy(self):
        return self.accuracy
    
    def set_accuracy(self, new_accuracy):
        self.accuracy = new_accuracy 
    
    def get_recall(self):
        return self.recall
    
    def set_recall(self, new_recall):
        self.recall = new_recall 
    
    def get_false_positive(self):
        return self.false_positive
    
    def set_false_positive(self, new_false_positive):
        self.recall = new_false_positive 