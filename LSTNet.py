class LSTNet(nn.Module):
    
    def __init__(self):
        super(LSTNet, self).__init__()
        self.num_features = 5
        self.conv1_out_channels = 32 
        self.conv1_kernel_height = 7
        self.recc1_out_channels = 64 
        self.skip_steps = [4, 24] 
        self.skip_reccs_out_channels = [4, 4] 
        self.output_out_features = 1
        self.ar_window_size = 7
        self.dropout = nn.Dropout(p = 0.2)
       
        
        self.conv1 = nn.Conv2d(1, self.conv1_out_channels, 
                               kernel_size=(self.conv1_kernel_height, self.num_features))
        self.recc1 = nn.GRU(self.conv1_out_channels, self.recc1_out_channels, batch_first=True)
        self.skip_reccs = {}
        for i in range(len(self.skip_steps)):
            self.skip_reccs[i] = nn.GRU(self.conv1_out_channels, self.skip_reccs_out_channels[i], batch_first=True)
        self.output_in_features = self.recc1_out_channels + np.dot(self.skip_steps, self.skip_reccs_out_channels)
        self.output = nn.Linear(self.output_in_features, self.output_out_features)
        if self.ar_window_size > 0:
            self.ar = nn.Linear(self.ar_window_size, 1)
        
    def forward(self, X):
        """
        Parameters:
        X (tensor) [batch_size, time_steps, num_features]
        """
        batch_size = X.size(0)
        
        # Convolutional Layer
        C = X.unsqueeze(1) # [batch_size, num_channels=1, time_steps, num_features]
        C = F.relu(self.conv1(C)) # [batch_size, conv1_out_channels, shrinked_time_steps, 1]
        C = self.dropout(C)
        C = torch.squeeze(C, 3) # [batch_size, conv1_out_channels, shrinked_time_steps]
        
        # Recurrent Layer
        R = C.permute(0, 2, 1) # [batch_size, shrinked_time_steps, conv1_out_channels]
        out, hidden = self.recc1(R) # [batch_size, shrinked_time_steps, recc_out_channels]
        R = out[:, -1, :] # [batch_size, recc_out_channels]
        R = self.dropout(R)
        #print(R.shape)
        
        # Skip Recurrent Layers
        shrinked_time_steps = C.size(2)
        for i in range(len(self.skip_steps)):
            skip_step = self.skip_steps[i]
            skip_sequence_len = shrinked_time_steps // skip_step
            # shrinked_time_steps shrinked further
            S = C[:, :, -skip_sequence_len*skip_step:] # [batch_size, conv1_out_channels, shrinked_time_steps]
            S = S.view(S.size(0), S.size(1), skip_sequence_len, skip_step) # [batch_size, conv1_out_channels, skip_sequence_len, skip_step=num_skip_components]
            # note that num_skip_components = skip_step
            S = S.permute(0, 3, 2, 1).contiguous() # [batch_size, skip_step=num_skip_components, skip_sequence_len, conv1_out_channels]
            S = S.view(S.size(0)*S.size(1), S.size(2), S.size(3))  # [batch_size*num_skip_components, skip_sequence_len, conv1_out_channels]
            out, hidden = self.skip_reccs[i](S) # [batch_size*num_skip_components, skip_sequence_len, skip_reccs_out_channels[i]]
            S = out[:, -1, :] # [batch_size*num_skip_components, skip_reccs_out_channels[i]]
            S = S.view(batch_size, skip_step*S.size(1)) # [batch_size, num_skip_components*skip_reccs_out_channels[i]]
            S = self.dropout(S)
            R = torch.cat((R, S), 1) # [batch_size, recc_out_channels + skip_reccs_out_channels * num_skip_components]
            #print(S.shape)
        #print(R.shape)
        
        # Output Layer
        O = F.relu(self.output(R)) # [batch_size, output_out_features=1]
        
        if self.ar_window_size > 0:
            # set dim3 based on output_out_features
            AR = X[:, -self.ar_window_size:, 3:4] # [batch_size, ar_window_size, output_out_features=1]
            AR = AR.permute(0, 2, 1).contiguous() # [batch_size, output_out_features, ar_window_size]
            AR = self.ar(AR) # [batch_size, output_out_features, 1]
            AR = AR.squeeze(2) # [batch_size, output_out_features]
            O = O + AR
        
        return O