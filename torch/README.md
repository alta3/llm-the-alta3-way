

```bash
sudo apt install cuda-drivers --upgrade
# reboot
{
  python3 -m venv torch/venv/
  source torch/venv/bin/activate
  python3 -m pip install -r torch/requirements.txt
}
```
