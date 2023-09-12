### Running Kubernettes Cluster

#### Setting Up Docker
* Navigate to CAMP-RT/
* Build docker image with image name camprt
> docker build . -t <name of docker image> camprt
* Todo: figure out what this does
> docker image tag camprt iridium.evl.uic.edu:5000camprt
* run it and then exit without closing image.  Make any changes to repo here.
> docker run -it camprt
> \^p^q
* check container name with
> docker ps
> docker commit (container name) iridium.uic.edu:5000/camprt
* I am not sure if this is necessary

> docker attach (container name)
> exit
* push it 
> docker push iridium.evl.uic.edu:5000/camprt
#### Starting Cluster
 > ssh <username>@k8slogin.evl.uic.edu -p 2222
 * Navigate to CAMP-RT/docker
 > kubectl apply -f <Name of yaml file> camprnettes.yaml
 > kubectl exec -it <Name of pod in file> camprt  -- /bin/bash
 
#### Running notebooks inside cluster
* Navigate to CAMP-RT/docker
* (If not already done):
> chmod +x ./run_scripts.sh
* Edit run scripts .sh to call
> python3 name_of_notebook.py
* Converts all ipynb files to py files and runs the above script
> ./run_scripts  
