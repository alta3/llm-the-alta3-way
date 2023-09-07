{
  sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin no/g' /etc/ssh/sshd_config
  sed -i 's/%sudo\sALL=(ALL:ALL) ALL/%sudo   ALL=(ALL) NOPASSWD:ALL/g' /etc/sudoers
  groupadd ubuntu
  useradd -m -g ubuntu -G sudo -s /usr/bin/bash ubuntu
  mkdir -p /home/ubuntu/.ssh
  cat /root/.ssh/authorized_keys > /home/ubuntu/.ssh/authorized_keys
  chown -R ubuntu:ubuntu /home/ubuntu/.ssh 
  systemctl restart ssh
}
