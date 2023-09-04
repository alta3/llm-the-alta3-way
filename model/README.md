## Models Directory Layout

```
└─── model_name_30B
    ├── clean.sh -> ../clean.sh  // cleanup script, deletes .bin files
    ├── fetch.sh -> ../fetch.sh  // fetch model data (see Aria2 lists)
    ├── unsplit.sh               // unsplit large models if required
    ├── README.md                // the original repo's model card
    ├── TheBloke_README.md       // TheBloke quntization model card
    └── model.list               // List of files to fetch
```

### Aria2 lists

Model parameter files are provided in each directory with the file name `<model>.aria2.list`.
To use these files manually follow this example:

```bash
cd <model>
aria2c --input-file <model>.aria2.list -x 16 -j 16 -s 16 -c
```
