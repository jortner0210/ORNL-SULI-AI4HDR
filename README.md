## AI4HDR-IR Prototype Use:
    - Setup Environment:
        conda create --name <env_name>
        conda activate <env_name>
        pip install -r requirements.txt

    - Initialize a database with 'initDB' script located in the query-system-prototype/backend directory.
    
    - Launch application: defaults - host="localhost", port=8000
        python app.py --db_loc=<database-location>

    - To access from another machine on the same network, establish ssh tunnel: 
        ssh -N -f -L localhost:8080:localhost:8000 <username>@cusg0 

    - Using a web browser, navigate to: localhost:8080
    

## CAEModel Use
    - Initialize:
        model = HDRClusterEncoder(
            init_model=True,  // Set to False, if loading model from disk
            lr=0.001,         // Autoencoder learning rate
            latent_dim=2048,  // Autoencoder latent space layer size
            n_clusters=10,    // Number of clusters in 
            activation="elu", // Model activation function
            stride=stride     // Autoencoder stride
        )

    - Compile:
        model.compile(
            loss=['kld', 'mse'], // Loss function for cluster model and autoencoder respectively
            loss_weights=[1, 1]  // How to weight the loss functions in training
        )

    - Pretrain: Trains the Autoencoder only
        model.pretrain(
            x,                   // Training data 
            batch_size=1,        // Data batch size
            epochs=1,            // Training number of epochs
            validation_split=0.0 // Training/Validation split
        )

    - Cluster Train: Joint training of Autoencoder and Cluster Layer
        model.clusterTrain(
            data,          // Training data
            batch_size,    // Training batch size
            checkpoint_dir // Directory to save model checkpoints
        )


## Other Notes:
    
    - To install pymeanshift: pip install git+git://github.com/fjean/pymeanshift.git


## Acknowledgement:
    This work was supported in part by the U.S. Department of Energy, Office of Science, Office of Workforce Development for Teachers and Scientists (WDTS) under the Science Undergraduate Laboratory Internships Program (SULI) program at Oak Ridge National Laboratory, administered by the Oak Ridge Institute for Science and Education.