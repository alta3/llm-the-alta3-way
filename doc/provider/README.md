# Providers

Searching for GPU's via this [great spreadsheet](https://fullstackdeeplearning.com/cloud-gpus/).

### Quicklinks

- [shadeform.ai](https://platform.shadeform.ai/)
- [datacrunch.io](https://cloud.datacrunch.io/dashboard/deploy-server/)
- [oblivus.com](https://console.oblivus.com/dashboard/oblivuscloud/deploy/)
- [vultr.com](https://my.vultr.com/deploy/)
- [paperspace.com](https://console.paperspace.com/bryfry/machines/create)

Specific setups and scripts for navigating the cloud gpu landscape

### [Paperspace.com](./paperspace/README.md)

- [Provider specific setup](./paperspace/setup.sh)
- [Home Page](https://www.paperspace.com/)
- Ubuntu 22.04 image
- GPUs:
  - A100 SXM4 80GB
  - A100 SXM4 40GB
- Observed Bandwidth: `~285Mbps`
- Expected download times:
  | model      | size  | download time | 
  |------------|------:|---------------|
  | llama 70b  | 188GB | ~1.5 hours    |
  | orca 13b   | 33GB  | ~15 minutes   |
  | falcon 40b | 78GB  | ~35 minutes   |

### [DataCrunch.io](./datacrunch/README.md)

- [Provider specific setup](./datacrunch/setup.sh)
- [Home Page](https://datacrunch.io)
- Ubuntu 22.04 image
- GPUs:
  - A100 SXM4 80GB
  - A100 SXM4 40GB

### [Lambda Labs](https://cloud.lambdalabs.com)

- Ubuntu 20.04 image
- GPUs 
  - A100 (40 GB PCIe)
  - A100 (40 GB SXM4)


### Oblivus

[Availability Page](https://console.oblivus.com/dashboard/availability/gpu/)
