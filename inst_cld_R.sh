#!/bin/bash


## installing latest R on ec2 or Azure. 
## Create a swap file to reduce the likelihood of virtual memory exhaustion on micro.
## source: http://danielgriff.in/ and also 
##  Andrew Collier on http://www.exegetic.biz/blog/2015/06/amazon-ec2-upgrading-r/

free -m #to check how much memory we have
sudo dd if=/dev/zero of=/swapfile bs=1M count=512

sudo mkswap /swapfile
sudo swapon /swapfile
sudo chown root:root /swapfile
sudo chmod 0600 /swapfile

### Add CRAN repo
echo "adding CRAN repo for latest R ..."
sudo add-apt-repository "deb https://cloud.r-project.org/bin/linux/ubuntu xenial-cran35/"

## Add key for secure apt (probably not need)
echo "adding keys for R..."
gpg --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys E084DAB9
gpg -a --export E084DAB9 | sudo apt-key add -

# update package list
sudo apt-get update
sudo apt-get install r-base
#need to confirm (could be made a bash function if too repetitive)
echo "Check if input required.. and enter 'yes' or 'no' "
read input
echo "input $input recorded. Proceeding..."
echo $input

sudo apt-get install cmake #required for light gbm e.g.
#need to confirm
echo "Check if input required.. and enter 'yes' or 'no' "
read input
echo $input

# dependencies, libraries for curl, xml
sudo apt-get install build-essential libcurl4-gnutls-dev libxml2-dev libssl-dev    

#install devtools so kernel can be installed for Jupyter
# Jupyter must already be installed!
R -e "install.packages(c('devtools'), repos='http://cran.rstudio.com/')"
R -e "devtools::install_github('IRkernel/IRkernel')"
R -e "IRkernel::installspec()"
	
# other packages
R -e "devtools::install_github('Microsoft/LightGBM', subdir = 'R-package')"

#yet other packages 
cat << EOF | sudo R --slave
install.packages(c(
  "tidyverse",
  "caret",
  "xgboost",
  "grf",
  ggplot2
))
EOF

#when done with installation could turn swapping off
#sudo swapoff /swapfile
