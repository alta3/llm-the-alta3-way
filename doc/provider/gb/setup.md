Acces the machine with a custom [ssh_config.rw](https://github.com/alta3/infrastructure/blob/main/charlie/ssh_config.rw).

1. Set the ssh config to access the *ssh_config.rw* file in infrastructure as follows:

    `ln -sf $HOME/git/infrastructure/charlie/ssh_config.rw $HOME/.ssh/config`

0. ssh to the enchilada host.

    `ssh enchilada.charlie`
