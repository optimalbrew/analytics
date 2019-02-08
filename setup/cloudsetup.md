## Apache Spark, Jupyter and R Setup on AWS EC2 or Azure

For Azure, the SSH creds need to be generated locally. Good to store them in a designated directory. 
	
	ssh-keygen -t rsa #for Azure not AWS

when prompted, enter file name: say, myAzKey and leaving passphrase blank is okay.
Two files will be created: *myAzKey* and the public one *myAzKey.pub* to be used for 
access as necessary.

On AWS *they* generate the keys and we download the .pem file on to the local machine. More
 details on SSH or scp on [AWS docs](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/AccessingInstancesLinux.html)
including list of _default usernames_ for various linux AMIs.

If the key is recently generated (they expire) then make sure the new one (downloaded from AWS)
has the private permissions
	
	chmod 400 key_name.pem


### Get a VM instance running

* Request spot instances with Ubuntu (some red hat commands are different, e.g. apt-get is ubuntu).
* Pick some names for instances, groups and security (allow SSH, and later HTTPS for server)

### For SSH access 
On Azure, have to pick own username. Copy and paste SHH creds stored earlier from local 
machine to the request. Print the SSH public key from whichever directly it was saved on.

	cat myAzKey.pub  

Obtain the public IP for the VM, and then can connect to it via SSH using the username
and the **private** ssh key (from appropriate directory)
 	
	ssh -i myAzKey own_username@publicIPofVM

For AWS use the key provided during account set up, or request new one. Save it e.g. *pair1.pem*  
**AWS keys** are region specific, and may not show up in other regions. Be sure to pick the 
correct zone to use pre-existing keys.

Also, on AWS EC2 username depends on linux version, e.g. Amazon linux it is ec-2user and
on Ubuntu, the username to SSH with is *ubuntu@*


		ssh -i pair1.pem ubuntu@whatever_public_DNS_is


* verify and then launch. For multiple instances, jot down which one will be master, and which ones slaves. 
Could have as many terminals open as nodes (or use split panes as on iTerm2, a terminal emulator with lots of goodies).

		sudo apt-get update #freshen things up a bit

### Virtual Memory Swapping 
This is really to reduce the chance of running out of virtual memory while installing 
R packages (later on). Should not be necessary with instances with RAM large enough to 
handle data entirely.

	free -m #to check how much memory we have
	sudo dd if=/dev/zero of=/swapfile bs=1M count=512
	#sudo dd if=/dev/zero of=/swapfile bs=1G count=2 
	
	sudo mkswap /swapfile
	sudo swapon /swapfile
	sudo chown root:root /swapfile
	sudo chmod 0600 /swapfile


  
### Check java 
Usually not installed, but available
	
		$ java -version
	
should have the run time and sdk installed, if not, then


	sudo apt-get install default-jdk  # run time is part of it
	
### install scala and pip
	
	sudo apt-get install scala
	
	sudo apt install python-pip	

	

### Get spark

	$ wget http://www-us.apache.org/dist/spark/spark-2.3.1/spark-2.3.1-bin-hadoop2.7.tgz -P ~/Downloads 
	$ sudo tar xvf ~/Downloads/spark-* -C /usr/local 
	$ sudo mv /usr/local/spark-* /usr/local/spark	
	  
Get pyspark (this usually fails on Azure, for small instance memory anyway)

	sudo pip install pyspark #see below for Azure

If "memory error" then try (without cache directory)

	sudo pip --no-cache-dir install pyspark


Add spark environment vars to profile (use nano.profile)

	$ nano .profile

enter the text and save the file

	export SPARK_HOME=/usr/local/spark
	export PATH=$PATH:$SPARK_HOME/bin
	
source it (in bash and some other shells)

	allnodes$ source ~/.profile

or . with a space which is for all shells
	
	. ~/.profile
	

change ownership to user ubuntu (default username on AWS) use own
username on Azure .. but why is this required? Doesn't seem to matter.

	$ sudo chown -R ubuntu $SPARK_HOME
	

Common spark configuration (from template)

	$ cp $SPARK_HOME/conf/spark-env.sh.template $SPARK_HOME/conf/spark-env.sh
	

spark-env.sh use nano to modify

	nano $SPARK_HOME/conf/spark-env.sh

cluster version (why is export DNS?, I have those commented out)

	export JAVA_HOME=/usr
	#export SPARK_PUBLIC_DNS=”current_node_public_dns”
	#export SPARK_WORKER_CORES=2

### Multiple nodes anyone?

create a slaves file (can be omitted with single VM)

	spark_master_node$ touch $SPARK_HOME/conf/slaves
	
In that file $SPARK_HOME/conf/slaves (use nano)
	
	spark_worker1_public_dns	#2nd if needed
	spark_worker2_public_dns	#3rd if needed
	.
	.
	.
	
### Start the node
	
Start the master node

	spark_master_node$ $SPARK_HOME/sbin/start-all.sh

It complains, about permission denied. But is often running (sometimes explicitly).  

Install and check to see if pyspark runs
	pip install pyspark
	pyspark

Check further by running examples
	
	cd $SPARK_HOME
	python examples/src/main/python/pi.py 10

or 

	./bin/spark-submit examples/src/main/python/pi.py 20


If that does not work, try creatig SSH creds

	ssh-keygen -t rsa

enter some folder under /home/spark or something else if that is denied

	 /home/spark/.ssh/id_rsa

but keep passphrase empty

	cp /home/spark/.ssh/id_rsa.pub .ssh/authorized_keys
	ssh localhost
	start-all.sh 

	
Can go to spark_master_public_dns:8080 to see worker nodes online. If you can't see them, 
check security settings to allow all traffic not just ssh
 		

### Shutting the node (or just terminate)	
	spark_master_node$ $SPARK_HOME/sbin/stop-all.sh


## Ipython/Jupyter notebooks

As ubuntu comes with python, getting jupyter to work is mostly about installing Anaconda. 
One option is to pick a community AMI on EC2, and make sure it is an 
[officially recognized](https://docs.anaconda.com/anaconda/user-guide/tasks/integration/amazon-aws#ca-images) 
version of Anaconda3 (or 2)

To install to an existing VM instance use same instructions [as for linux](https://docs.anaconda.com/anaconda/install/linux)
As of writing the current version of the installer is at
https://repo.anaconda.com/archive/Anaconda3-5.2.0-Linux-x86_64.sh

	wget https://repo.anaconda.com/archive/Anaconda3-5.2.0-Linux-x86_64.sh -P ~/Downloads

Then  use (even if bash is not the shell) to install conda for python 3.6
	
	cd Downloads
	bash Anaconda3-5.2.0-Linux-x86_64.sh #takes a while, chill.
	cd

see further [installation details](https://docs.anaconda.com/anaconda/install/linux) (not a whole lot).
Don't really have to open the link. When the installer is done, either close and reopen 
the terminal. Or source it

	source ~/.bashrc


### Get certs and configure jupyter

From [AWS docs](https://docs.aws.amazon.com/mxnet/latest/dg/setup-jupyter-configure-server.html)
Create a SSL cert.

	cd #just to make sure we are at the top
	mkdir ssl
	cd ssl
	sudo openssl req -x509 -nodes -days 365 -newkey rsa:1024 -keyout "cert.key" -out "cert.pem" -batch 

Create a password to log into the Jupyter notebook server later. 

Open iPython terminal
	
	ipython

at the ipython promt, use the passwd() command to set password

	ipythonPrompt> from IPython.lib import passwd 
	ipythonPrompt> passwd()

**record** the password hash (e.g. sha1:ex123: ...)

	ipythonPrompt> exit

Create jupyter config. file

	jupyter notebook --generate-config 

The command creates a configuration file (jupyter_notebook_config.py) in the ~/.jupyter directory. 

Update the configuration file to store your password and SSL certificate information.

Edit the config file (assuming nano).

	nano	~/.jupyter/jupyter_notebook_config.py
	
Paste the following text at the **end** of the file (use ctrl-V to skip pages on nano). 
**NOTE: the following is for ubuntu on AWS**. For Azure, or for other instances, change the
username below from ubuntu in the two lines for /home/ **username**  /ssl

	c = get_config()  # Get the config object.
	c.NotebookApp.certfile = u'/home/ubuntu/ssl/cert.pem' # path to the certificate we generated
	c.NotebookApp.keyfile = u'/home/ubuntu/ssl/cert.key' # path to the certificate key we generated
	c.IPKernelApp.pylab = 'inline'  # in-line figure when using Matplotlib
	c.NotebookApp.ip = '*'  # Serve notebooks locally.
	c.NotebookApp.open_browser = False  # Do not open a browser window by default when using notebooks.
	c.NotebookApp.password = ' sha saved earlier ' 

save the file and exit.


## Install R
This may not be enough as some packages (ggpplot2, xgboost, grf) only work with latest 
versions of R (3.3.+). Would need to get binaries directly from CRAN? Or check what comes
with Anaconda's R essentials.
	
### Add CRAN repo
	sudo add-apt-repository "deb https://cloud.r-project.org/bin/linux/ubuntu xenial-cran35/"

## Add key for secure apt 
Not really needed, we trust CRAN repos. Anyway..

	gpg --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys E084DAB9
	gpg -a --export E084DAB9 | sudo apt-key add -
	
	sudo apt-get install r-base
	sudo apt-get install cmake # e.g. for lightgbm
	
Also need  dependencies (packages for curl, xml) before we install devtools on ubuntu.	
	
	sudo apt-get install build-essential libcurl4-gnutls-dev libxml2-dev libssl-dev	


	
### Install R packages. 
Install devtools	
	R -e "install.packages('devtools')"

Install the Jupyter kernel

	R -e "devtools::install_github('IRkernel/IRkernel')"
	R -e "IRkernel::installspec()"
	
Install other stuff as needed (or later)

	R -e "install.packages(c('ggplot2', 'tidyverse', 'grf', 'xgboost'))"
	R -e "devtools::install_github('Microsoft/LightGBM', subdir = 'R-package')"

## Set up port forwarding
Forward port on local machine, use the public DNS of instance (master). Assuming ubuntu on 
AWS, change for others. This is just like the initial SSH into the instance. 
	
	$ ssh -i pair2.pem -L 8157:127.0.0.1:8888 ubuntu@ 

This command opens a tunnel between local client and the remote EC2 instance that is running
 Jupyter server. After running the command, we can access the Jupyter notebook server at 
https://127.0.0.1:8157.


Create a directory for storing Jupyter notebooks.

	$ mkdir ~/mynotebooks
	$ cd ~/mynotebooks

Start the Jupyter notebook server.

	$ jupyter notebook

By default, the server runs on port 8888 (see above). If that port is not available, Jupyter uses the 
next available port. The Jupyter terminal shows the port on which the server is listening.

### Connect to the notebook server


In the address bar of your browser, enter the URL:

    https://127.0.0.1:8157

If the connection is successful,  we'll see the Jupyter notebook server home page. 
Type the password that created when configuring the Jupyter server. 

### Working with S3 storage
First get the AWS's command line interface
	pip install awscli

to list all S3 folders (that the AIM has been given access to). This should list the top 
dir name.   
	
	aws s3 ls s3://
	
More detail (for example the bucket pcarkaggle/s3) 
		
	aws s3 ls s3://pcarkaggle/s3/

Moving files (reverse order as required).
	
	aws s3 cp s3://pcarkaggle/s3/test_5k.csv ~/data/test_5klocal.csv
