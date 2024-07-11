Acces the machine with 
`ssh paperspace@a100-1.alta3.training`
Assuming the routing in AWS hasnt changed. 

{
  sudo sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin no/g' /etc/ssh/sshd_config
  sudo sed -i 's/%sudo\sALL=(ALL:ALL) ALL/%sudo   ALL=(ALL) NOPASSWD:ALL/g' /etc/sudoers
  sudo groupadd ubuntu
  sudo useradd -m -g ubuntu -G sudo -s /usr/bin/bash ubuntu
  sudo mkdir -p /home/ubuntu/.ssh
  cat ~/.ssh/authorized_keys | sudo tee /home/ubuntu/.ssh/authorized_keys
  sudo chown -R ubuntu:ubuntu /home/ubuntu/.ssh 
  sudo systemctl restart ssh
}
