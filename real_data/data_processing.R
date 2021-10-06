install.packages("plotly")
install.packages("BiocManager")
if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")

BiocManager::install("genefu")

library(breastCancerMAINZ)
library(breastCancerVDX)
library(breastCancerTRANSBIG)
library(Biobase)

library(genefu)

data(mainz)
data(vdx)
data(transbig)

table(pData(transbig)$series)
table(pData(vdx)$series)

intersect(pData(transbig)$filename, pData(vdx)$filename)
intersect(pData(transbig)$id, pData(vdx)$id)


transbig.subt = molecular.subtyping(sbt.model="pam50", data=t(exprs(transbig)),annot=featureData(transbig)@data, do.mapping=TRUE, verbose=TRUE)
table(transbig.subt$subtype)

vdx.subt = molecular.subtyping(sbt.model="pam50", data=t(exprs(vdx)),annot=featureData(vdx)@data, do.mapping=TRUE)
table(vdx.subt$subtype)

mainz.subt = molecular.subtyping(sbt.model="pam50", data=t(exprs(mainz)),annot=featureData(mainz)@data, do.mapping=TRUE)
table(mainz.subt$subtype)


transbig.subt2 = molecular.subtyping(sbt.model="scmgene", data=t(exprs(transbig)),annot=featureData(transbig)@data, do.mapping=TRUE, verbose=TRUE)
table(transbig.subt2$subtype)

vdx.subt2 = molecular.subtyping(sbt.model="scmgene", data=t(exprs(vdx)),annot=featureData(vdx)@data, do.mapping=TRUE)
table(vdx.subt2$subtype)

mainz.subt2 = molecular.subtyping(sbt.model="scmgene", data=t(exprs(mainz)),annot=featureData(mainz)@data, do.mapping=TRUE)
table(mainz.subt2$subtype)


subs = c("ER-/HER2-" ="Basal", "HER2+"="Her2", "ER+/HER2- High Prolif" = "LumB", "ER+/HER2- Low Prolif"="LumA" )


transbig_ = data.frame(PAM50 = transbig.subt$subtype, HK_3GENE = transbig.subt2$subtype, stringsAsFactors=FALSE)
transbig_$HK_3GENE = subs[transbig_$HK_3GENE]


vdx_ = data.frame(PAM50 = vdx.subt$subtype, HK_3GENE = vdx.subt2$subtype, stringsAsFactors=FALSE)
vdx_$HK_3GENE = subs[vdx_$HK_3GENE]


mainz_ = data.frame(PAM50 = mainz.subt$subtype, HK_3GENE = mainz.subt2$subtype, stringsAsFactors=FALSE)
mainz_$HK_3GENE = subs[mainz_$HK_3GENE]

write.csv(mainz_, file="../Dropbox (MIT)/clust_stuff_exp/paper_plots/Real_Data/Data/mainz_subtypes.csv")
write.csv(transbig_, file= "../Dropbox (MIT)/clust_stuff_exp/paper_plots/Real_Data/Data/transbig_subtypes.csv")
write.csv(vdx_, file="../Dropbox (MIT)/clust_stuff_exp/paper_plots/Real_Data/Data/vdx_subtypes.csv")

### 3 gene

library(plotly)

genes = c("205225_at", "216836_s_at",  "208079_s_at")


mainz_3 = t(exprs(mainz)[genes,])
trans_3 = t(exprs(transbig)[genes,])
vdx_3 = t(exprs(vdx)[genes,])

#write.csv(mainz_3, file="Dropbox (MIT)/clust_stuff_exp/paper_plots/Real_Data/Data/MAINZ_3_SV")
#write.csv(trans_3, file="Dropbox (MIT)/clust_stuff_exp/paper_plots/Real_Data/Data/TRANS_3_SV")
#write.csv(vdx_3, file="Dropbox (MIT)/clust_stuff_exp/paper_plots/Real_Data/Data/VDX_3_SV")

write.csv(mainz_3, file = '../Dropbox (MIT)/clust_stuff_exp/paper_plots/Real_Data/Data/MAINZ_3_SV.csv')
write.csv(trans_3, file = '../Dropbox (MIT)/clust_stuff_exp/paper_plots/Real_Data/Data/TRANS_3_SV.csv')
write.csv(vdx_3, file = '../Dropbox (MIT)/clust_stuff_exp/paper_plots/Real_Data/Data/VDX_3_SV.csv')



write.csv(mainz_3, file = 'Dropbox (MIT)/Rep.clustering/github_code/real_data/r_data/MAINZ_3_SV.csv')
write.csv(trans_3, file = 'Dropbox (MIT)/Rep.clustering/github_code/real_data/r_data/TRANS_3_SV.csv')
write.csv(vdx_3, file = 'Dropbox (MIT)/Rep.clustering/github_code/real_data/r_data/VDX_3_SV.csv')

all.equal(rownames(mainz_3), rownames(mainz_))

plot_ly(x=mainz_3[,"205225_at"], y=mainz_3[,"216836_s_at"],  z=mainz_3[,"208079_s_at"], type="scatter3d", mode="markers" ,  color= mainz_$HK_3GENE, main= "Mainz" )
#plot_ly(x=mainz_3[,"205225_at"], y=mainz_3[,"216836_s_at"],  z=mainz_3[,"208079_s_at"], type="scatter3d", mode="markers" ,  color= mainz_$PAM50, main= "Mainz" )

plot_ly(x=trans_3[,"205225_at"], y=trans_3[,"216836_s_at"],  z=trans_3[,"208079_s_at"], type="scatter3d", mode="markers" ,  color= transbig_$HK_3GENE, main= "Transbig" )
plot_ly(x=vdx_3[,"205225_at"], y=vdx_3[,"216836_s_at"],  z=vdx_3[,"208079_s_at"], type="scatter3d", mode="markers" ,  color=vdx_$HK_3GENE, main= "Mainz" )



## PAM50 on PCA

genes = rownames(subset(featureData(mainz)@data, EntrezGene.ID%in%pam50$centroids.map$EntrezGene.ID))
mainz_e = exprs(mainz)[genes,]
write.csv(mainz_e, file = '../Dropbox (MIT)/clust_stuff_exp/paper_plots/Real_Data/Data/MAINZ_PAM_SV.csv')

genes = rownames(subset(featureData(transbig)@data, EntrezGene.ID%in%pam50$centroids.map$EntrezGene.ID))
trans_e = exprs(transbig)[genes,]
write.csv(trans_e, file = '../Dropbox (MIT)/clust_stuff_exp/paper_plots/Real_Data/Data/TRANS_PAM_SV.csv')

genes = rownames(subset(featureData(vdx)@data, EntrezGene.ID%in%pam50$centroids.map$EntrezGene.ID))
vdx_e = exprs(vdx)[genes,]
write.csv(vdx_e, file = '../Dropbox (MIT)/clust_stuff_exp/paper_plots/Real_Data/Data/VDX_PAM_SV.csv')


dim(mainz_e) # why 90 genes?
write.csv(mainz_e, file = '../Dropbox (MIT)/clust_stuff_exp/paper_plots/Real_Data/Data/MAINZ_PAM_SV.csv')
#cvs = apply(mainz_e, 1, function(x){sd(x)/mean(x)})
all.equal(colnames(mainz_e), rownames(mainz_))

all.equal(rownames(mainz_e), rownames(vdx_e))
#mainz_e = mainz_e[order(cvs, decreasing=TRUE)[1:223],]

pca = prcomp(t(mainz_e), retx=TRUE)

PC = data.frame(pca$x[,c(1:5)])
PC$HK_3GENE = mainz_$HK_3GENE
PC$PAM50=mainz_$PAM50

#head(PC)

plot_ly(x=PC$PC1, y=PC$PC2,  z=PC$PC3, type="scatter3d", mode="markers" ,  color=PC$HK_3GENE, main= "Mainz" )




mainz_e = exprs(transbig)[genes,]


all.equal(colnames(mainz_e), rownames(transbig_))
pca = prcomp(t(mainz_e), retx=TRUE)

PC = data.frame(pca$x[,c(1:5)])
PC$HK_3GENE = transbig_$HK_3GENE
PC$PAM50=transbig_$PAM50

#head(PC)

plot_ly(x=PC$PC1, y=PC$PC2,  z=PC$PC3, type="scatter3d", mode="markers" ,  color=PC$HK_3GENE, main= "Transbig" )





mainz_e = exprs(vdx)[genes,]


all.equal(colnames(mainz_e), rownames(vdx_))

pca = prcomp(t(mainz_e), retx=TRUE)

PC = data.frame(pca$x[,c(1:5)])
PC$HK_3GENE = vdx_$HK_3GENE
PC$PAM50=vdx_$PAM50

#head(PC)

plot_ly(x=PC$PC1, y=PC$PC2,  z=PC$PC3, type="scatter3d", mode="markers" ,  color=PC$HK_3GENE, main= "VDX" )






## Top variable genes

mainz_e = exprs(mainz)

cvs = apply(mainz_e, 1, function(x){sd(x)/mean(x)})

all.equal(colnames(mainz_e), rownames(mainz_))
mainz_e = mainz_e[order(cvs, decreasing=TRUE)[1:223],]

pca = prcomp(t(mainz_e), retx=TRUE)

PC = data.frame(pca$x[,c(1:5)])
PC$HK_3GENE = mainz_$HK_3GENE
PC$PAM50=mainz_$PAM50

#head(PC)

plot_ly(x=PC$PC1, y=PC$PC2,  z=PC$PC3, type="scatter3d", mode="markers" ,  color=PC$HK_3GENE, main= "Mainz" )




mainz_e = exprs(transbig)

cvs = apply(mainz_e, 1, function(x){sd(x)/mean(x)})

all.equal(colnames(mainz_e), rownames(transbig_))
mainz_e = mainz_e[order(cvs, decreasing=TRUE)[1:223],]

pca = prcomp(t(mainz_e), retx=TRUE)

PC = data.frame(pca$x[,c(1:5)])
PC$HK_3GENE = transbig_$HK_3GENE
PC$PAM50=transbig_$PAM50

#head(PC)

plot_ly(x=PC$PC1, y=PC$PC2,  z=PC$PC3, type="scatter3d", mode="markers" ,  color=PC$HK_3GENE, main= "Transbig" )





mainz_e = exprs(vdx)

cvs = apply(mainz_e, 1, function(x){sd(x)/mean(x)})

all.equal(colnames(mainz_e), rownames(vdx_))
mainz_e = mainz_e[order(cvs, decreasing=TRUE)[1:223],]

pca = prcomp(t(mainz_e), retx=TRUE)

PC = data.frame(pca$x[,c(1:5)])
PC$HK_3GENE = vdx_$HK_3GENE
PC$PAM50=vdx_$PAM50

#head(PC)

plot_ly(x=PC$PC1, y=PC$PC2,  z=PC$PC3, type="scatter3d", mode="markers" ,  color=PC$HK_3GENE, main= "VDX" )





ggplot(PC, aes(x=PC1, y=PC2, colour=HK_3GENE ))+geom_point(size=2.5) +theme(panel.background = element_rect(fill = 'white'), panel.grid.major = element_line(colour = "lightgray"),panel.grid.minor = element_line(colour = "gray"))

ggplot(PC, aes(x=PC2, y=PC3, colour=Batch,shape=ROI ))+geom_point(size=2.5) +theme(panel.background = element_rect(fill = 'white'), panel.grid.major = element_line(colour = "lightgray"),panel.grid.minor = element_line(colour = "gray"))

ggplot(PC, aes(x=PC1, y=PC3, colour=Batch,shape=ROI ))+geom_point(size=2.5) +theme(panel.background = element_rect(fill = 'white'), panel.grid.major = element_line(colour = "lightgray"),panel.grid.minor = element_line(colour = "gray"))
