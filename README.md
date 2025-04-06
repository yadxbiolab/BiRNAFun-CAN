# BiRNAFun-CAN

## Table of Contents
- [Test Dataset Descriptions](#data-test)
- [How-to-run-BiRNAFun-CAN?](#how-to-run-BiRNAFun-CAN)



### Test Dataset Descriptions (stored in this repository)

#### 1. bifunctional RNA Test Dataset
***File path***: `BiRNAFun-CAN/data/test-.fasta`
***File path***: `BiRNAFun-CAN/data/test+.fasta`
***Correct directory***:  
                        BiFunRNA-CAN
                           data
                              test-.fasta
                              test+.fasta
                           Features
                              DNACodingDist
                              Full_step1.txt
                              Full_step3.txt
                              orf_step1.txt
                              orf_step3.txt
                              embedding
                              mirna_doc2vec.model
                              mirna_doc2vec.model.dv.vectors.npy
                              __init__.py
                              additional_features.py
                              BaseClass.py
                              CTD.py
                              EIIP.py
                              Fickett.py
                              FrameKmer.py
                              GCcontent.py
                              kmer3.py
                              MFE.py
                              ORF.py
                              ORF_Length.py
                              ProtParam.py
                              train2vec.py
                              word2vec_feature.py
                           Generate2.py
                           model.pth
                           Model.py
                           scaler_mean.npy
                           scaler_scale.npy


### How to run BiRNAFun-CAN
#### 1. Input File
Prepare your input file, which should be in fasta format. For example, the file name could be `test.fasta`.

#### 2. Output File
Define a name for your output file, which will be used to store the results processed by the script. For example, you might name your output file `predicted_score.excel`.

#### 3. Download this repository and open the Command Line Interface ####
   Open a terminal (Linux ) or VScode (Windows).

#### 4. Navigate to the Directory Containing the Script ####
   Use the `cd` command to navigate to the directory where your script is saved. For example:
   ```bash
   cd BiRNAFun-CAN
   ```

#### 5. Run the Script ####
   Use the following command to run the script, replacing <input_file> with your actual file paths.
   ```bash
   python Model.py <input_file> 
   ```
   For example:
   ```bash
   python Model.py ../data/test-.fasta 
   ```

#### 6. Example Output ####
   The output file will contain the identifier of each sequence, the calculated score (the probility of bifunctional RNA), and the sequence itself. For example:
   ```python
   >Sequence1  0.233736  CCTTCTTGCTCTATT...
   >Sequence2  0.442661  CAAAGTGCTGGGATT...
    ...
   ```




