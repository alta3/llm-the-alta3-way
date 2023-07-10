### Aria2 lists

Model parameter files are provided in each directory with the file name `<model>.aria2.list`.
To use these files manually follow this example:

```bash
cd <model>
aria2c --input-file <model>.aria2.list -x 16 -j 16 -s 16 -c
```
