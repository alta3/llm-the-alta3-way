# DataCrunch.io

### Setup ssh access

- Disable root access 
- Setup ubuntu user with same authorized keys
- Make sudeoers passwordless

```bash
ssh {{ ip_or_fqdn }} -l root
```

```bash
setup.sh
```

```bash
ssh {{ ip_or_fqdn }} -l ubuntu
```
