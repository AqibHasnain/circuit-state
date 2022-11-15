library("regutools")


## Connect to the RegulonDB database if necessary
regulondb_conn <- connect_database()

#> snapshotDate(): 2021-08-03
#> adding rname 'https://www.dropbox.com/s/ufp6wqcv5211v1w/regulondb_v10.8_sqlite.db?dl=1'

## Build a regulondb object
e_coli_regulondb <-
  regulondb(
    database_conn = regulondb_conn,
    organism = "E.coli",
    database_version = "1",
    genome_version = "1"
  )

## Get the araC regulators
araC_regulation <-
  get_gene_regulators(
    e_coli_regulondb,
    genes = c("araC"),
    format = "multirow",
    output.type = "TF"
  )

## Summarize the araC regulation
get_regulatory_summary(e_coli_regulondb, araC_regulation)

## Get the TF-Gene Regulatory Network
tf_gene_network <- get_regulatory_network(e_coli_regulondb,type="TF-GENE") # other types: 'GENE-GENE', 'TF-TF'


df_tf_gene <- as.data.frame(tf_gene_network)

## write to csv
write.csv(df_tf_gene, file="TF-Gene_network_Ecoli_K12.csv")

