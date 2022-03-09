


library('RcppCNPy')
library(reshape2)
library(ggplot2)
library(rjson)



#setwd( "C:/Users/pabgon/cnn_pso/NEW_CODES_2107_ph_indep_cens")
#setwd( "C:/Users/pabgon/project_2/multiple_times_sim_results/case1_N1000")

# N 1000

#case 1
#po_so
setwd( "C:/Users/pabgon/project_2/POCNN_so/case1_N1000")


train_loss_po_so <- npyLoad("train_loss_po_so.npy")
val_loss_po_so <- npyLoad("val_loss_po_so.npy")

apply(train_loss_po_so,1,mean)
apply(val_loss_po_so,1,mean)

case1_po_so = data.frame(cbind(apply(train_loss_po_so,1,mean),apply(val_loss_po_so,1,mean)))
#labels = c("coxCNN","poCNN1","poCNN2","ipcw_poCNN1","ipcw_poCNN2")
labels = c("train","val")
colnames(case1_po_so)<-labels
case1_po_so$id <- seq_len(nrow(case1_po_so))
case1_po_so_time1 <- reshape2::melt(case1_po_so, id.vars = c('id'))
# rename
names(case1_po_so_time1) <- c('id', 'split', 'loss')


p <- ggplot(case1_po_so_time1, aes(id, loss,fill=factor(split), colour=split))
p+geom_line() + labs(x="Epochs",y="Loss") + theme(panel.grid.major = element_blank() ) +
  ggtitle("POCNN single output, Case 1, N=1000")



#po_mo
setwd( "C:/Users/pabgon/project_2/POCNN_mo/case1_N1000")


train_loss_po_mo <- npyLoad("train_loss_po_mlt.npy")
val_loss_po_mo <- npyLoad("val_loss_po_mlt.npy")

case1_po_mo = data.frame(cbind(apply(train_loss_po_mo,1,mean),apply(val_loss_po_mo,1,mean)))
labels = c("train","val")
colnames(case1_po_mo)<-labels
case1_po_mo$id <- seq_len(nrow(case1_po_mo))
case1_po_mo_time1 <- reshape2::melt(case1_po_mo, id.vars = c('id'))
# rename
names(case1_po_mo_time1) <- c('id', 'split', 'loss')


p <- ggplot(case1_po_mo_time1, aes(id, loss,fill=factor(split), colour=split))
p+geom_line()+ labs(x="Epochs",y="Loss") + theme(panel.grid.major = element_blank() ) +
  ggtitle("POCNN multi-output, Case 1, N=1000")

#cox
setwd( "C:/Users/pabgon/project_2/COX_CNN/case1_N1000")

train_loss_cox <- npyLoad("train_loss_cox.npy")
val_loss_cox <- npyLoad("val_loss_cox.npy")

case1_cox = data.frame(cbind(apply(train_loss_cox,1,mean),apply(val_loss_cox,1,mean)))
labels = c("train","val")
colnames(case1_cox)<-labels
case1_cox$id <- seq_len(nrow(case1_cox))
case1_cox_time1 <- reshape2::melt(case1_cox, id.vars = c('id'))
# rename
names(case1_cox_time1) <- c('id', 'split', 'loss')


p <- ggplot(case1_cox_time1, aes(id, loss,fill=factor(split), colour=split))
p+geom_line()+ labs(x="Epochs",y="Loss") + theme(panel.grid.major = element_blank() ) +
  ggtitle("CoxCNN, Case 1, N=1000")

#po_ipcw

setwd( "C:/Users/pabgon/project_2/IPCW_POCNN_so/case1_N1000")

train_loss_ipcwpo_so <- npyLoad("train_loss_ipcwpo_so.npy")
val_loss_ipcwpo_so <- npyLoad("val_loss_ipcwpo_so.npy")

case1_ipcwpo_so = data.frame(cbind(apply(train_loss_ipcwpo_so,1,mean),apply(val_loss_ipcwpo_so,1,mean)))
labels = c("train","val")
colnames(case1_ipcwpo_so)<-labels
case1_ipcwpo_so$id <- seq_len(nrow(case1_ipcwpo_so))
case1_ipcwpo_so_time1 <- reshape2::melt(case1_ipcwpo_so, id.vars = c('id'))
# rename
names(case1_ipcwpo_so_time1) <- c('id', 'split', 'loss')


p <- ggplot(case1_ipcwpo_so_time1, aes(id, loss,fill=factor(split), colour=split))
p+geom_line()+ labs(x="Epochs",y="Loss") + theme(panel.grid.major = element_blank() ) +
  ggtitle("IPCW-POCNN single output, Case 1, N=1000")

#po_mtl_ipcw

setwd( "C:/Users/pabgon/project_2/IPCW_POCNN_mo/case1_N1000")

train_loss_ipcwpo_mo <- npyLoad("train_loss_ipcwpo_mlt.npy")
val_loss_ipcwpo_mo <- npyLoad("val_loss_ipcwpo_mlt.npy")

case1_ipcwpo_mo = data.frame(cbind(apply(train_loss_ipcwpo_mo,1,mean),apply(val_loss_ipcwpo_mo,1,mean)))
labels = c("train","val")
colnames(case1_ipcwpo_mo)<-labels
case1_ipcwpo_mo$id <- seq_len(nrow(case1_ipcwpo_mo))
case1_ipcwpo_mo_time1 <- reshape2::melt(case1_ipcwpo_mo, id.vars = c('id'))
# rename
names(case1_ipcwpo_mo_time1) <- c('id', 'split', 'loss')


p <- ggplot(case1_ipcwpo_mo_time1, aes(id, loss,fill=factor(split), colour=split))
p+geom_line()+ labs(x="Epochs",y="Loss") + theme(panel.grid.major = element_blank() ) +
  ggtitle("IPCW-POCNN multi-output, Case 1, N=1000")

#### CASE 2 ####

#po_so
setwd( "C:/Users/pabgon/project_2/POCNN_so/case2_N1000")

train_loss_po_so <- npyLoad("train_loss_po_so.npy")
val_loss_po_so <- npyLoad("val_loss_po_so.npy")

case2_po_so = data.frame(cbind(apply(train_loss_po_so,1,mean),apply(val_loss_po_so,1,mean)))
#labels = c("coxCNN","poCNN1","poCNN2","ipcw_poCNN1","ipcw_poCNN2")
labels = c("train","val")
colnames(case2_po_so)<-labels
case2_po_so$id <- seq_len(nrow(case2_po_so))
case2_po_so_time1 <- reshape2::melt(case2_po_so, id.vars = c('id'))
# rename
names(case2_po_so_time1) <- c('id', 'split', 'loss')


p <- ggplot(case2_po_so_time1, aes(id, loss,fill=factor(split), colour=split))
p+geom_line() + labs(x="Epochs",y="Loss") + theme(panel.grid.major = element_blank() ) +
  ggtitle("POCNN single output, Case 2, N=1000")

#po_mo
setwd( "C:/Users/pabgon/project_2/POCNN_mo/case2_N1000")

train_loss_po_mo <- npyLoad("train_loss_po_mlt.npy")
val_loss_po_mo <- npyLoad("val_loss_po_mlt.npy")

case2_po_mo = data.frame(cbind(apply(train_loss_po_mo,1,mean),apply(val_loss_po_mo,1,mean)))
labels = c("train","val")
colnames(case2_po_mo)<-labels
case2_po_mo$id <- seq_len(nrow(case2_po_mo))
case2_po_mo_time1 <- reshape2::melt(case2_po_mo, id.vars = c('id'))
# rename
names(case2_po_mo_time1) <- c('id', 'split', 'loss')


p <- ggplot(case2_po_mo_time1, aes(id, loss,fill=factor(split), colour=split))
p+geom_line()+ labs(x="Epochs",y="Loss") + theme(panel.grid.major = element_blank() ) +
  ggtitle("POCNN multi-output, Case 2, N=1000")

#cox
setwd( "C:/Users/pabgon/project_2/COX_CNN/case2_N1000")

train_loss_cox <- npyLoad("train_loss_cox.npy")
val_loss_cox <- npyLoad("val_loss_cox.npy")

case2_cox = data.frame(cbind(apply(train_loss_cox,1,mean),apply(val_loss_cox,1,mean)))
labels = c("train","val")
colnames(case2_cox)<-labels
case2_cox$id <- seq_len(nrow(case2_cox))
case2_cox_time1 <- reshape2::melt(case2_cox, id.vars = c('id'))
# rename
names(case2_cox_time1) <- c('id', 'split', 'loss')


p <- ggplot(case2_cox_time1, aes(id, loss,fill=factor(split), colour=split))
p+geom_line()+ labs(x="Epochs",y="Loss") + theme(panel.grid.major = element_blank() ) +
  ggtitle("CoxCNN, Case 2, N=1000")

#po_ipcw

setwd( "C:/Users/pabgon/project_2/IPCW_POCNN_so/case2_N1000")

train_loss_ipcwpo_so <- npyLoad("train_loss_ipcwpo_so.npy")
val_loss_ipcwpo_so <- npyLoad("val_loss_ipcwpo_so.npy")

case2_ipcwpo_so = data.frame(cbind(apply(train_loss_ipcwpo_so,1,mean),apply(val_loss_ipcwpo_so,1,mean)))
labels = c("train","val")
colnames(case2_ipcwpo_so)<-labels
case2_ipcwpo_so$id <- seq_len(nrow(case2_ipcwpo_so))
case2_ipcwpo_so_time1 <- reshape2::melt(case2_ipcwpo_so, id.vars = c('id'))
# rename
names(case2_ipcwpo_so_time1) <- c('id', 'split', 'loss')


p <- ggplot(case2_ipcwpo_so_time1, aes(id, loss,fill=factor(split), colour=split))
p+geom_line()+ labs(x="Epochs",y="Loss") + theme(panel.grid.major = element_blank() ) +
  ggtitle("IPCW-POCNN single output, Case 2, N=1000")

#po_mtl_ipcw
setwd( "C:/Users/pabgon/project_2/IPCW_POCNN_mo/case2_N1000")

train_loss_ipcwpo_mo <- npyLoad("train_loss_ipcwpo_mlt.npy")
val_loss_ipcwpo_mo <- npyLoad("val_loss_ipcwpo_mlt.npy")

case2_ipcwpo_mo = data.frame(cbind(apply(train_loss_ipcwpo_mo,1,mean),apply(val_loss_ipcwpo_mo,1,mean)))
labels = c("train","val")
colnames(case2_ipcwpo_mo)<-labels
case2_ipcwpo_mo$id <- seq_len(nrow(case2_ipcwpo_mo))
case2_ipcwpo_mo_time1 <- reshape2::melt(case2_ipcwpo_mo, id.vars = c('id'))
# rename
names(case2_ipcwpo_mo_time1) <- c('id', 'split', 'loss')


p <- ggplot(case2_ipcwpo_mo_time1, aes(id, loss,fill=factor(split), colour=split))
p+geom_line()+ labs(x="Epochs",y="Loss") + theme(panel.grid.major = element_blank() ) +
  ggtitle("IPCW-POCNN multi-output, Case 2, N=1000")
#### CASE 3 ####

#po_so
setwd( "C:/Users/pabgon/project_2/POCNN_so/case3_N1000")


train_loss_po_so <- npyLoad("train_loss_po_so.npy")
val_loss_po_so <- npyLoad("val_loss_po_so.npy")

case3_po_so = data.frame(cbind(apply(train_loss_po_so,1,mean),apply(val_loss_po_so,1,mean)))
#labels = c("coxCNN","poCNN1","poCNN2","ipcw_poCNN1","ipcw_poCNN2")
labels = c("train","val")
colnames(case3_po_so)<-labels
case3_po_so$id <- seq_len(nrow(case3_po_so))
case3_po_so_time1 <- reshape2::melt(case3_po_so, id.vars = c('id'))
# rename
names(case3_po_so_time1) <- c('id', 'split', 'loss')


p <- ggplot(case3_po_so_time1, aes(id, loss,fill=factor(split), colour=split))
p+geom_line() + labs(x="Epochs",y="Loss") + theme(panel.grid.major = element_blank() ) +
  ggtitle("POCNN single output, Case 3, N=1000")

#po_mo
setwd( "C:/Users/pabgon/project_2/POCNN_mo/case3_N1000")


train_loss_po_mo <- npyLoad("train_loss_po_mlt.npy")
val_loss_po_mo <- npyLoad("val_loss_po_mlt.npy")

case3_po_mo = data.frame(cbind(apply(train_loss_po_mo,1,mean),apply(val_loss_po_mo,1,mean)))
labels = c("train","val")
colnames(case3_po_mo)<-labels
case3_po_mo$id <- seq_len(nrow(case3_po_mo))
case3_po_mo_time1 <- reshape2::melt(case3_po_mo, id.vars = c('id'))
# rename
names(case3_po_mo_time1) <- c('id', 'split', 'loss')


p <- ggplot(case3_po_mo_time1, aes(id, loss,fill=factor(split), colour=split))
p+geom_line()+ labs(x="Epochs",y="Loss") + theme(panel.grid.major = element_blank() ) +
  ggtitle("POCNN multi-output, Case 3, N=1000")

#cox
setwd( "C:/Users/pabgon/project_2/COX_CNN/case3_N1000")


train_loss_cox <- npyLoad("train_loss_cox.npy")
val_loss_cox <- npyLoad("val_loss_cox.npy")

case3_cox = data.frame(cbind(apply(train_loss_cox,1,mean),apply(val_loss_cox,1,mean)))
labels = c("train","val")
colnames(case3_cox)<-labels
case3_cox$id <- seq_len(nrow(case3_cox))
case3_cox_time1 <- reshape2::melt(case3_cox, id.vars = c('id'))
# rename
names(case3_cox_time1) <- c('id', 'split', 'loss')


p <- ggplot(case3_cox_time1, aes(id, loss,fill=factor(split), colour=split))
p+geom_line()+ labs(x="Epochs",y="Loss") + theme(panel.grid.major = element_blank() ) +
  ggtitle("CoxCNN, Case 3, N=1000")

#po_ipcw
setwd( "C:/Users/pabgon/project_2/IPCW_POCNN_so/case3_N1000")

train_loss_ipcwpo_so <- npyLoad("train_loss_ipcwpo_so.npy")
val_loss_ipcwpo_so <- npyLoad("val_loss_ipcwpo_so.npy")

case3_ipcwpo_so = data.frame(cbind(apply(train_loss_ipcwpo_so,1,mean),apply(val_loss_ipcwpo_so,1,mean)))
labels = c("train","val")
colnames(case3_ipcwpo_so)<-labels
case3_ipcwpo_so$id <- seq_len(nrow(case3_ipcwpo_so))
case3_ipcwpo_so_time1 <- reshape2::melt(case3_ipcwpo_so, id.vars = c('id'))
# rename
names(case3_ipcwpo_so_time1) <- c('id', 'split', 'loss')


p <- ggplot(case3_ipcwpo_so_time1, aes(id, loss,fill=factor(split), colour=split))
p+geom_line()+ labs(x="Epochs",y="Loss") + theme(panel.grid.major = element_blank() ) +
  ggtitle("IPCW-POCNN single output, Case 3, N=1000")

#po_mtl_ipcw
setwd( "C:/Users/pabgon/project_2/IPCW_POCNN_mo/case3_N1000")

train_loss_ipcwpo_mo <- npyLoad("train_loss_ipcwpo_mlt.npy")
val_loss_ipcwpo_mo <- npyLoad("val_loss_ipcwpo_mlt.npy")

case3_ipcwpo_mo = data.frame(cbind(apply(train_loss_ipcwpo_mo,1,mean),apply(val_loss_ipcwpo_mo,1,mean)))
labels = c("train","val")
colnames(case3_ipcwpo_mo)<-labels
case3_ipcwpo_mo$id <- seq_len(nrow(case3_ipcwpo_mo))
case3_ipcwpo_mo_time1 <- reshape2::melt(case3_ipcwpo_mo, id.vars = c('id'))
# rename
names(case3_ipcwpo_mo_time1) <- c('id', 'split', 'loss')


p <- ggplot(case3_ipcwpo_mo_time1, aes(id, loss,fill=factor(split), colour=split))
p+geom_line()+ labs(x="Epochs",y="Loss") + theme(panel.grid.major = element_blank() ) +
  ggtitle("IPCW-POCNN multi-output, Case 3, N=1000")


#### CASE 4 ####

#po_so
setwd( "C:/Users/pabgon/project_2/POCNN_so/case4_N1000")


train_loss_po_so <- npyLoad("train_loss_po_so.npy")
val_loss_po_so <- npyLoad("val_loss_po_so.npy")

case4_po_so = data.frame(cbind(apply(train_loss_po_so,1,mean),apply(val_loss_po_so,1,mean)))
#labels = c("coxCNN","poCNN1","poCNN2","ipcw_poCNN1","ipcw_poCNN2")
labels = c("train","val")
colnames(case4_po_so)<-labels
case4_po_so$id <- seq_len(nrow(case4_po_so))
case4_po_so_time1 <- reshape2::melt(case4_po_so, id.vars = c('id'))
# rename
names(case4_po_so_time1) <- c('id', 'split', 'loss')


p <- ggplot(case4_po_so_time1, aes(id, loss,fill=factor(split), colour=split))
p+geom_line() + labs(x="Epochs",y="Loss") + theme(panel.grid.major = element_blank() ) +
  ggtitle("POCNN single output, Case 4, N=1000")

#po_mo
setwd( "C:/Users/pabgon/project_2/POCNN_mo/case4_N1000")


train_loss_po_mo <- npyLoad("train_loss_po_mlt.npy")
val_loss_po_mo <- npyLoad("val_loss_po_mlt.npy")

case4_po_mo = data.frame(cbind(apply(train_loss_po_mo,1,mean),apply(val_loss_po_mo,1,mean)))
labels = c("train","val")
colnames(case4_po_mo)<-labels
case4_po_mo$id <- seq_len(nrow(case4_po_mo))
case4_po_mo_time1 <- reshape2::melt(case4_po_mo, id.vars = c('id'))
# rename
names(case4_po_mo_time1) <- c('id', 'split', 'loss')


p <- ggplot(case4_po_mo_time1, aes(id, loss,fill=factor(split), colour=split))
p+geom_line()+ labs(x="Epochs",y="Loss") + theme(panel.grid.major = element_blank() ) +
  ggtitle("POCNN multi-output, Case 4, N=1000")

#cox
setwd( "C:/Users/pabgon/project_2/COX_CNN/case4_N1000")


train_loss_cox <- npyLoad("train_loss_cox.npy")
val_loss_cox <- npyLoad("val_loss_cox.npy")

case4_cox = data.frame(cbind(apply(train_loss_cox,1,mean),apply(val_loss_cox,1,mean)))
labels = c("train","val")
colnames(case4_cox)<-labels
case4_cox$id <- seq_len(nrow(case4_cox))
case4_cox_time1 <- reshape2::melt(case4_cox, id.vars = c('id'))
# rename
names(case4_cox_time1) <- c('id', 'split', 'loss')


p <- ggplot(case4_cox_time1, aes(id, loss,fill=factor(split), colour=split))
p+geom_line()+ labs(x="Epochs",y="Loss") + theme(panel.grid.major = element_blank() ) +
  ggtitle("CoxCNN, Case 4, N=1000")

#po_ipcw
setwd( "C:/Users/pabgon/project_2/IPCW_POCNN_so/case4_N1000")

train_loss_ipcwpo_so <- npyLoad("train_loss_ipcwpo_so.npy")
val_loss_ipcwpo_so <- npyLoad("val_loss_ipcwpo_so.npy")

case4_ipcwpo_so = data.frame(cbind(apply(train_loss_ipcwpo_so,1,mean),apply(val_loss_ipcwpo_so,1,mean)))
labels = c("train","val")
colnames(case4_ipcwpo_so)<-labels
case4_ipcwpo_so$id <- seq_len(nrow(case4_ipcwpo_so))
case4_ipcwpo_so_time1 <- reshape2::melt(case4_ipcwpo_so, id.vars = c('id'))
# rename
names(case4_ipcwpo_so_time1) <- c('id', 'split', 'loss')


p <- ggplot(case4_ipcwpo_so_time1, aes(id, loss,fill=factor(split), colour=split))
p+geom_line()+ labs(x="Epochs",y="Loss") + theme(panel.grid.major = element_blank() ) +
  ggtitle("IPCW-POCNN single output, Case 4, N=1000")

#po_mtl_ipcw
setwd( "C:/Users/pabgon/project_2/IPCW_POCNN_mo/case4_N1000")

train_loss_ipcwpo_mo <- npyLoad("train_loss_ipcwpo_mlt.npy")
val_loss_ipcwpo_mo <- npyLoad("val_loss_ipcwpo_mlt.npy")

case4_ipcwpo_mo = data.frame(cbind(apply(train_loss_ipcwpo_mo,1,mean),apply(val_loss_ipcwpo_mo,1,mean)))
labels = c("train","val")
colnames(case4_ipcwpo_mo)<-labels
case4_ipcwpo_mo$id <- seq_len(nrow(case4_ipcwpo_mo))
case4_ipcwpo_mo_time1 <- reshape2::melt(case4_ipcwpo_mo, id.vars = c('id'))
# rename
names(case4_ipcwpo_mo_time1) <- c('id', 'split', 'loss')


p <- ggplot(case4_ipcwpo_mo_time1, aes(id, loss,fill=factor(split), colour=split))
p+geom_line()+ labs(x="Epochs",y="Loss") + theme(panel.grid.major = element_blank() ) +
  ggtitle("IPCW-POCNN multi-output, Case 4, N=1000")

#### CASE 5 ####

#po_so
setwd( "C:/Users/pabgon/project_2/POCNN_so/case5_N1000")


train_loss_po_so <- npyLoad("train_loss_po_so.npy")
val_loss_po_so <- npyLoad("val_loss_po_so.npy")

case5_po_so = data.frame(cbind(apply(train_loss_po_so,1,mean),apply(val_loss_po_so,1,mean)))
#labels = c("coxCNN","poCNN1","poCNN2","ipcw_poCNN1","ipcw_poCNN2")
labels = c("train","val")
colnames(case5_po_so)<-labels
case5_po_so$id <- seq_len(nrow(case5_po_so))
case5_po_so_time1 <- reshape2::melt(case5_po_so, id.vars = c('id'))
# rename
names(case5_po_so_time1) <- c('id', 'split', 'loss')


p <- ggplot(case5_po_so_time1, aes(id, loss,fill=factor(split), colour=split))
p+geom_line() + labs(x="Epochs",y="Loss") + theme(panel.grid.major = element_blank() ) +
  ggtitle("POCNN single output, Case 5, N=1000")

#po_mo
setwd( "C:/Users/pabgon/project_2/POCNN_mo/case5_N1000")


train_loss_po_mo <- npyLoad("train_loss_po_mlt.npy")
val_loss_po_mo <- npyLoad("val_loss_po_mlt.npy")

case5_po_mo = data.frame(cbind(apply(train_loss_po_mo,1,mean),apply(val_loss_po_mo,1,mean)))
labels = c("train","val")
colnames(case5_po_mo)<-labels
case5_po_mo$id <- seq_len(nrow(case5_po_mo))
case5_po_mo_time1 <- reshape2::melt(case5_po_mo, id.vars = c('id'))
# rename
names(case5_po_mo_time1) <- c('id', 'split', 'loss')


p <- ggplot(case5_po_mo_time1, aes(id, loss,fill=factor(split), colour=split))
p+geom_line()+ labs(x="Epochs",y="Loss") + theme(panel.grid.major = element_blank() ) +
  ggtitle("POCNN multi-output, Case 5, N=1000")


#cox
setwd( "C:/Users/pabgon/project_2/COX_CNN/case5_N1000")


train_loss_cox <- npyLoad("train_loss_cox.npy")
val_loss_cox <- npyLoad("val_loss_cox.npy")

case5_cox = data.frame(cbind(apply(train_loss_cox,1,mean),apply(val_loss_cox,1,mean)))
labels = c("train","val")
colnames(case5_cox)<-labels
case5_cox$id <- seq_len(nrow(case5_cox))
case5_cox_time1 <- reshape2::melt(case5_cox, id.vars = c('id'))
# rename
names(case5_cox_time1) <- c('id', 'split', 'loss')


p <- ggplot(case5_cox_time1, aes(id, loss,fill=factor(split), colour=split))
p+geom_line()+ labs(x="Epochs",y="Loss") + theme(panel.grid.major = element_blank() ) +
  ggtitle("CoxCNN, Case 5, N=1000")

#po_ipcw
setwd( "C:/Users/pabgon/project_2/IPCW_POCNN_so/case5_N1000")

train_loss_ipcwpo_so <- npyLoad("train_loss_ipcwpo_so.npy")
val_loss_ipcwpo_so <- npyLoad("val_loss_ipcwpo_so.npy")

case5_ipcwpo_so = data.frame(cbind(apply(train_loss_ipcwpo_so,1,mean),apply(val_loss_ipcwpo_so,1,mean)))
labels = c("train","val")
colnames(case5_ipcwpo_so)<-labels
case5_ipcwpo_so$id <- seq_len(nrow(case5_ipcwpo_so))
case5_ipcwpo_so_time1 <- reshape2::melt(case5_ipcwpo_so, id.vars = c('id'))
# rename
names(case5_ipcwpo_so_time1) <- c('id', 'split', 'loss')


p <- ggplot(case5_ipcwpo_so_time1, aes(id, loss,fill=factor(split), colour=split))
p+geom_line()+ labs(x="Epochs",y="Loss") + theme(panel.grid.major = element_blank() ) +
  ggtitle("IPCW-POCNN single output, Case 5, N=1000")

#po_mtl_ipcw
setwd( "C:/Users/pabgon/project_2/IPCW_POCNN_mo/case5_N1000")

train_loss_ipcwpo_mo <- npyLoad("train_loss_ipcwpo_mlt.npy")
val_loss_ipcwpo_mo <- npyLoad("val_loss_ipcwpo_mlt.npy")

case5_ipcwpo_mo = data.frame(cbind(apply(train_loss_ipcwpo_mo,1,mean),apply(val_loss_ipcwpo_mo,1,mean)))
labels = c("train","val")
colnames(case5_ipcwpo_mo)<-labels
case5_ipcwpo_mo$id <- seq_len(nrow(case5_ipcwpo_mo))
case5_ipcwpo_mo_time1 <- reshape2::melt(case5_ipcwpo_mo, id.vars = c('id'))
# rename
names(case5_ipcwpo_mo_time1) <- c('id', 'split', 'loss')


p <- ggplot(case5_ipcwpo_mo_time1, aes(id, loss,fill=factor(split), colour=split))
p+geom_line()+ labs(x="Epochs",y="Loss") + theme(panel.grid.major = element_blank() ) +
  ggtitle("IPCW-POCNN multi-output, Case 5, N=1000")

#### CASE 6 ####

#po_so
setwd( "C:/Users/pabgon/project_2/POCNN_so/case6_N1000")


train_loss_po_so <- npyLoad("train_loss_po_so.npy")
val_loss_po_so <- npyLoad("val_loss_po_so.npy")

case6_po_so = data.frame(cbind(apply(train_loss_po_so,1,mean),apply(val_loss_po_so,1,mean)))
#labels = c("coxCNN","poCNN1","poCNN2","ipcw_poCNN1","ipcw_poCNN2")
labels = c("train","val")
colnames(case6_po_so)<-labels
case6_po_so$id <- seq_len(nrow(case6_po_so))
case6_po_so_time1 <- reshape2::melt(case6_po_so, id.vars = c('id'))
# rename
names(case6_po_so_time1) <- c('id', 'split', 'loss')


p <- ggplot(case6_po_so_time1, aes(id, loss,fill=factor(split), colour=split))
p+geom_line() + labs(x="Epochs",y="Loss") + theme(panel.grid.major = element_blank() ) +
  ggtitle("POCNN single output, Case 6, N=1000")

#po_mo
setwd( "C:/Users/pabgon/project_2/POCNN_mo/case6_N1000")


train_loss_po_mo <- npyLoad("train_loss_po_mlt.npy")
val_loss_po_mo <- npyLoad("val_loss_po_mlt.npy")

case6_po_mo = data.frame(cbind(apply(train_loss_po_mo,1,mean),apply(val_loss_po_mo,1,mean)))
labels = c("train","val")
colnames(case6_po_mo)<-labels
case6_po_mo$id <- seq_len(nrow(case6_po_mo))
case6_po_mo_time1 <- reshape2::melt(case6_po_mo, id.vars = c('id'))
# rename
names(case6_po_mo_time1) <- c('id', 'split', 'loss')


p <- ggplot(case6_po_mo_time1, aes(id, loss,fill=factor(split), colour=split))
p+geom_line()+ labs(x="Epochs",y="Loss") + theme(panel.grid.major = element_blank() ) +
  ggtitle("POCNN multi-output, Case 6, N=1000")

#cox
setwd( "C:/Users/pabgon/project_2/COX_CNN/case6_N1000")


train_loss_cox <- npyLoad("train_loss_cox.npy")
val_loss_cox <- npyLoad("val_loss_cox.npy")

case6_cox = data.frame(cbind(apply(train_loss_cox,1,mean),apply(val_loss_cox,1,mean)))
labels = c("train","val")
colnames(case6_cox)<-labels
case6_cox$id <- seq_len(nrow(case6_cox))
case6_cox_time1 <- reshape2::melt(case6_cox, id.vars = c('id'))
# rename
names(case6_cox_time1) <- c('id', 'split', 'loss')


p <- ggplot(case6_cox_time1, aes(id, loss,fill=factor(split), colour=split))
p+geom_line()+ labs(x="Epochs",y="Loss") + theme(panel.grid.major = element_blank() ) +
  ggtitle("CoxCNN, Case 6, N=1000")

#po_ipcw
setwd( "C:/Users/pabgon/project_2/IPCW_POCNN_so/case6_N1000")

train_loss_ipcwpo_so <- npyLoad("train_loss_ipcwpo_so.npy")
val_loss_ipcwpo_so <- npyLoad("val_loss_ipcwpo_so.npy")

case6_ipcwpo_so = data.frame(cbind(apply(train_loss_ipcwpo_so,1,mean),apply(val_loss_ipcwpo_so,1,mean)))
labels = c("train","val")
colnames(case6_ipcwpo_so)<-labels
case6_ipcwpo_so$id <- seq_len(nrow(case6_ipcwpo_so))
case6_ipcwpo_so_time1 <- reshape2::melt(case6_ipcwpo_so, id.vars = c('id'))
# rename
names(case6_ipcwpo_so_time1) <- c('id', 'split', 'loss')


p <- ggplot(case6_ipcwpo_so_time1, aes(id, loss,fill=factor(split), colour=split))
p+geom_line()+ labs(x="Epochs",y="Loss") + theme(panel.grid.major = element_blank() ) +
  ggtitle("IPCW-POCNN single output, Case 6, N=1000")

#po_mtl_ipcw
setwd( "C:/Users/pabgon/project_2/IPCW_POCNN_mo/case6_N1000")

train_loss_ipcwpo_mo <- npyLoad("train_loss_ipcwpo_mlt.npy")
val_loss_ipcwpo_mo <- npyLoad("val_loss_ipcwpo_mlt.npy")

case6_ipcwpo_mo = data.frame(cbind(apply(train_loss_ipcwpo_mo,1,mean),apply(val_loss_ipcwpo_mo,1,mean)))
labels = c("train","val")
colnames(case6_ipcwpo_mo)<-labels
case6_ipcwpo_mo$id <- seq_len(nrow(case6_ipcwpo_mo))
case6_ipcwpo_mo_time1 <- reshape2::melt(case6_ipcwpo_mo, id.vars = c('id'))
# rename
names(case6_ipcwpo_mo_time1) <- c('id', 'split', 'loss')


p <- ggplot(case6_ipcwpo_mo_time1, aes(id, loss,fill=factor(split), colour=split))
p+geom_line()+ labs(x="Epochs",y="Loss") + theme(panel.grid.major = element_blank() ) +
  ggtitle("IPCW-POCNN multi-output, Case 6, N=1000")

################################# A U  C   #############################

############################ plots violin #####################

library('RcppCNPy')
library(reshape2)
library(ggplot2)
library(rjson)



########  AUC   ###################
#case 1
#po_so
setwd( "C:/Users/pabgon/project_2/POCNN_so/case1_N1000")

auc1_po_so <- npyLoad("auc1_po_so.npy")
auc2_po_so <- npyLoad("auc2_po_so.npy")
auc3_po_so <- npyLoad("auc3_po_so.npy")
auc4_po_so <- npyLoad("auc4_po_so.npy")

auc_metric = data.frame(cbind(apply(auc1_po_so,1,mean),apply(auc2_po_so,1,mean),apply(auc3_po_so,1,mean),apply(auc4_po_so,1,mean),seq(1,25)))
colnames(auc_metric)<-c("auc1","auc2","auc3","auc4","epoch")


ggplot()+
  geom_line(data=auc_metric,aes(y=auc1,x= epoch,colour="time 1"),size=1 )+
  geom_line(data=auc_metric,aes(y=auc2,x= epoch,colour="time 2"),size=1) +
  geom_line(data=auc_metric,aes(y=auc3,x= epoch,colour="time 3"),size=1) +
  geom_line(data=auc_metric,aes(y=auc4,x= epoch,colour="time 4"),size=1) +
  scale_color_manual(name = "", values = c("time 1" = "blue", "time 2" = "red","time 3" = "green", "time 4" = "orange" )) +   labs(y=" AUC (test set) ") +
  ggtitle("POCNN single output, Case 1, N=1000")
  

#po_mo
setwd( "C:/Users/pabgon/project_2/POCNN_mo/case1_N1000")
auc1_po_mo <- npyLoad("auc1_po_mlt.npy")
auc2_po_mo <- npyLoad("auc2_po_mlt.npy")
auc3_po_mo <- npyLoad("auc3_po_mlt.npy")
auc4_po_mo <- npyLoad("auc4_po_mlt.npy")

auc_metric = data.frame(cbind(apply(auc1_po_mo,1,mean),apply(auc2_po_mo,1,mean),apply(auc3_po_mo,1,mean),apply(auc4_po_mo,1,mean),seq(1,25)))
colnames(auc_metric)<-c("auc1","auc2","auc3","auc4","epoch")


ggplot()+
  geom_line(data=auc_metric,aes(y=auc1,x= epoch,colour="time 1"),size=1 )+
  geom_line(data=auc_metric,aes(y=auc2,x= epoch,colour="time 2"),size=1) +
  geom_line(data=auc_metric,aes(y=auc3,x= epoch,colour="time 3"),size=1) +
  geom_line(data=auc_metric,aes(y=auc4,x= epoch,colour="time 4"),size=1) +
  scale_color_manual(name = "", values = c("time 1" = "blue", "time 2" = "red","time 3" = "green", "time 4" = "orange" )) +   labs(y=" AUC (test set) ") +
  ggtitle("POCNN multi-output, Case 1, N=1000")

#cox
setwd( "C:/Users/pabgon/project_2/COX_CNN/case1_N1000")

auc1_cox <- npyLoad(file = "auc1_cox.npy")
auc2_cox <- npyLoad(file = "auc2_cox.npy")
auc3_cox <- npyLoad(file = "auc3_cox.npy")
auc4_cox <- npyLoad(file = "auc4_cox.npy")

auc_metric = data.frame(cbind(apply(auc1_cox,1,mean),apply(auc2_cox,1,mean),apply(auc3_cox,1,mean),apply(auc4_cox,1,mean),seq(1,25)))
colnames(auc_metric)<-c("auc1","auc2","auc3","auc4","epoch")


ggplot()+
  geom_line(data=auc_metric,aes(y=auc1,x= epoch,colour="time 1"),size=1 )+
  geom_line(data=auc_metric,aes(y=auc2,x= epoch,colour="time 2"),size=1) +
  geom_line(data=auc_metric,aes(y=auc3,x= epoch,colour="time 3"),size=1) +
  geom_line(data=auc_metric,aes(y=auc4,x= epoch,colour="time 4"),size=1) +
  scale_color_manual(name = "", values = c("time 1" = "blue", "time 2" = "red","time 3" = "green", "time 4" = "orange" )) +   labs(y=" AUC (test set) ") +
  ggtitle("Cox-CNN, Case 1, N=1000")

#po_ipcw
setwd( "C:/Users/pabgon/project_2/IPCW_POCNN_so/case1_N1000")
auc1_ipcwpo_so <- npyLoad("auc1_ipcwpo_so.npy")
auc2_ipcwpo_so <- npyLoad("auc2_ipcwpo_so.npy")
auc3_ipcwpo_so <- npyLoad("auc3_ipcwpo_so.npy")
auc4_ipcwpo_so <- npyLoad("auc4_ipcwpo_so.npy")

auc_metric = data.frame(cbind(apply(auc1_ipcwpo_so,1,mean),apply(auc2_ipcwpo_so,1,mean),apply(auc3_ipcwpo_so,1,mean),apply(auc4_ipcwpo_so,1,mean),seq(1,25)))
colnames(auc_metric)<-c("auc1","auc2","auc3","auc4","epoch")


ggplot()+
  geom_line(data=auc_metric,aes(y=auc1,x= epoch,colour="time 1"),size=1 )+
  geom_line(data=auc_metric,aes(y=auc2,x= epoch,colour="time 2"),size=1) +
  geom_line(data=auc_metric,aes(y=auc3,x= epoch,colour="time 3"),size=1) +
  geom_line(data=auc_metric,aes(y=auc4,x= epoch,colour="time 4"),size=1) +
  scale_color_manual(name = "", values = c("time 1" = "blue", "time 2" = "red","time 3" = "green", "time 4" = "orange" )) +   labs(y=" AUC (test set) ") +
  ggtitle("IPCW-POCNN single output, Case 1, N=1000")

#po_mtl_ipcw
setwd( "C:/Users/pabgon/project_2/IPCW_POCNN_mo/case1_N1000")
auc1_ipcwpo_mo <- npyLoad("auc1_ipcwpo_mlt.npy")
auc2_ipcwpo_mo <- npyLoad("auc2_ipcwpo_mlt.npy")
auc3_ipcwpo_mo <- npyLoad("auc3_ipcwpo_mlt.npy")
auc4_ipcwpo_mo <- npyLoad("auc4_ipcwpo_mlt.npy")

auc_metric = data.frame(cbind(apply(auc1_ipcwpo_mo,1,mean),apply(auc2_ipcwpo_mo,1,mean),apply(auc3_ipcwpo_mo,1,mean),apply(auc4_ipcwpo_mo,1,mean),seq(1,25)))
colnames(auc_metric)<-c("auc1","auc2","auc3","auc4","epoch")


ggplot()+
  geom_line(data=auc_metric,aes(y=auc1,x= epoch,colour="time 1"),size=1 )+
  geom_line(data=auc_metric,aes(y=auc2,x= epoch,colour="time 2"),size=1) +
  geom_line(data=auc_metric,aes(y=auc3,x= epoch,colour="time 3"),size=1) +
  geom_line(data=auc_metric,aes(y=auc4,x= epoch,colour="time 4"),size=1) +
  scale_color_manual(name = "", values = c("time 1" = "blue", "time 2" = "red","time 3" = "green", "time 4" = "orange" )) +   labs(y=" AUC (test set) ") +
  ggtitle("IPCW-POCNN multi-output, Case 1, N=1000")


case1_auc1 = data.frame(cbind(auc1_cox[25,],auc1_po_so[25,], auc1_po_mo[25,],auc1_ipcwpo_so[25,],auc1_ipcwpo_mo[25,]))
#labels = c("coxCNN","poCNN1","poCNN2","ipcw_poCNN1","ipcw_poCNN2")
labels = c("cox","po1","po2","po3","po4")
colnames(case1_auc1)<-labels
case1_auc1$id <- seq_len(nrow(case1_auc1))
case1_time1 <- reshape2::melt(case1_auc1, id.vars = c('id'))
# rename
names(case1_time1) <- c('id', 'method', 'auc')
case1_time1$time = 'time 1'

case1_auc2 = data.frame(cbind(auc2_cox[25,],auc2_po_so[25,], auc2_po_mo[25,],auc2_ipcwpo_so[25,],auc2_ipcwpo_mo[25,]))
#labels = c("coxCNN","poCNN")
colnames(case1_auc2)<-labels
case1_auc2$id <- seq_len(nrow(case1_auc2))
case1_time2 <- reshape2::melt(case1_auc2, id.vars = c('id'))
# rename
names(case1_time2) <- c('id', 'method', 'auc')
case1_time2$time = 'time 2'

case1_auc3 = data.frame(cbind(auc3_cox[25,],auc3_po_so[25,], auc3_po_mo[25,],auc3_ipcwpo_so[25,],auc3_ipcwpo_mo[25,]))
#labels = c("coxCNN","poCNN")
colnames(case1_auc3)<-labels
case1_auc3$id <- seq_len(nrow(case1_auc3))
case1_time3 <- reshape2::melt(case1_auc3, id.vars = c('id'))
# rename
names(case1_time3) <- c('id', 'method', 'auc')
case1_time3$time = 'time 3'

case1_auc4 = data.frame(cbind(auc4_cox[25,],auc4_po_so[25,], auc4_po_mo[25,],auc4_ipcwpo_so[25,],auc4_ipcwpo_mo[25,]))
#labels = c("coxCNN","poCNN")
colnames(case1_auc4)<-labels
case1_auc4$id <- seq_len(nrow(case1_auc4))
case1_time4 <- reshape2::melt(case1_auc4, id.vars = c('id'))
# rename
names(case1_time4) <- c('id', 'method', 'auc')
case1_time4$time = 'time 4'


case1 = rbind(case1_time1,case1_time2,case1_time3, case1_time4)

### t test  ##

t.test(auc1_cox, auc1_po_so,paired=FALSE, conf.level=0.95)
t.test(auc1_cox, auc1_po_mo,paired=FALSE, conf.level=0.95)
t.test(auc1_cox, auc1_ipcwpo_so,paired=FALSE, conf.level=0.95)
t.test(auc1_cox, auc1_ipcwpo_mo,paired=FALSE, conf.level=0.95)

####### This is the plot
p <- ggplot(case1, aes(factor(method), auc,fill=time))
p +stat_boxplot(geom = "errorbar", width = 0.5) + geom_violin(trim=FALSE, fill='#A4A4A4', color="lightgrey",width=.7)+geom_boxplot(aes(fill = factor(method)),width=.3) +facet_wrap(~time,ncol = 5)+
  labs(x="",y="AUC")+ theme(legend.position = "none") +
  theme(panel.grid.major = element_blank(),
        axis.title.x = element_text( size=20),
        axis.title.y = element_text( size=20),
        axis.text.x = element_text( 
          size=15),
        axis.text.y = element_text(size=20)) +
  theme(strip.text.x = element_text(size = 15)) +
  theme(legend.position = "none")+ scale_y_continuous(breaks=seq(0,1,.05))

##############
ggplot(case1,aes(x=method, y=auc, fill=factor(time))) +
  geom_boxplot() + + geom_violin()
#############
p <- ggplot(case1, aes(factor(method), auc,fill=time))
p +stat_boxplot(geom = "errorbar", width = 0.5) +geom_boxplot(width=0.1) +facet_wrap(~time)+
  labs(x="",y="AUC")+ theme(panel.grid.major = element_blank(),
                            axis.title.x = element_text( size=20),
                            axis.title.y = element_text( size=20),
                            axis.text.x = element_text( 
                              size=15),
                            axis.text.y = element_text(size=20)) +
  theme(strip.text.x = element_text(size = 15)) +
  theme(legend.position = "none")+ scale_y_continuous(breaks=seq(0,1,.05))

#library(vioplot)
#vioplot(CASE1, ylab="AUC",col="lightgrey")
###################################################
#case 2

#po_so
setwd( "C:/Users/pabgon/project_2/POCNN_so/case2_N1000")

auc1_po_so <- npyLoad("auc1_po_so.npy")
auc2_po_so <- npyLoad("auc2_po_so.npy")
auc3_po_so <- npyLoad("auc3_po_so.npy")
auc4_po_so <- npyLoad("auc4_po_so.npy")

auc_metric = data.frame(cbind(apply(auc1_po_so,1,mean),apply(auc2_po_so,1,mean),apply(auc3_po_so,1,mean),apply(auc4_po_so,1,mean),seq(1,25)))
colnames(auc_metric)<-c("auc1","auc2","auc3","auc4","epoch")


ggplot()+
  geom_line(data=auc_metric,aes(y=auc1,x= epoch,colour="time 1"),size=1 )+
  geom_line(data=auc_metric,aes(y=auc2,x= epoch,colour="time 2"),size=1) +
  geom_line(data=auc_metric,aes(y=auc3,x= epoch,colour="time 3"),size=1) +
  geom_line(data=auc_metric,aes(y=auc4,x= epoch,colour="time 4"),size=1) +
  scale_color_manual(name = "", values = c("time 1" = "blue", "time 2" = "red","time 3" = "green", "time 4" = "orange" )) +   labs(y=" AUC (test set) ") +
  ggtitle("POCNN single output, Case 2, N=1000")

#po_mo
setwd( "C:/Users/pabgon/project_2/POCNN_mo/case2_N1000")
auc1_po_mo <- npyLoad("auc1_po_mlt.npy")
auc2_po_mo <- npyLoad("auc2_po_mlt.npy")
auc3_po_mo <- npyLoad("auc3_po_mlt.npy")
auc4_po_mo <- npyLoad("auc4_po_mlt.npy")

auc_metric = data.frame(cbind(apply(auc1_po_mo,1,mean),apply(auc2_po_mo,1,mean),apply(auc3_po_mo,1,mean),apply(auc4_po_mo,1,mean),seq(1,25)))
colnames(auc_metric)<-c("auc1","auc2","auc3","auc4","epoch")


ggplot()+
  geom_line(data=auc_metric,aes(y=auc1,x= epoch,colour="time 1"),size=1 )+
  geom_line(data=auc_metric,aes(y=auc2,x= epoch,colour="time 2"),size=1) +
  geom_line(data=auc_metric,aes(y=auc3,x= epoch,colour="time 3"),size=1) +
  geom_line(data=auc_metric,aes(y=auc4,x= epoch,colour="time 4"),size=1) +
  scale_color_manual(name = "", values = c("time 1" = "blue", "time 2" = "red","time 3" = "green", "time 4" = "orange" )) +   labs(y=" AUC (test set) ") +
  ggtitle("POCNN multi-output, Case 2, N=1000")

#cox
setwd( "C:/Users/pabgon/project_2/COX_CNN/case2_N1000")

auc1_cox <- npyLoad(file = "auc1_cox.npy")
auc2_cox <- npyLoad(file = "auc2_cox.npy")
auc3_cox <- npyLoad(file = "auc3_cox.npy")
auc4_cox <- npyLoad(file = "auc4_cox.npy")

auc_metric = data.frame(cbind(apply(auc1_cox,1,mean),apply(auc2_cox,1,mean),apply(auc3_cox,1,mean),apply(auc4_cox,1,mean),seq(1,25)))
colnames(auc_metric)<-c("auc1","auc2","auc3","auc4","epoch")


ggplot()+
  geom_line(data=auc_metric,aes(y=auc1,x= epoch,colour="time 1"),size=1 )+
  geom_line(data=auc_metric,aes(y=auc2,x= epoch,colour="time 2"),size=1) +
  geom_line(data=auc_metric,aes(y=auc3,x= epoch,colour="time 3"),size=1) +
  geom_line(data=auc_metric,aes(y=auc4,x= epoch,colour="time 4"),size=1) +
  scale_color_manual(name = "", values = c("time 1" = "blue", "time 2" = "red","time 3" = "green", "time 4" = "orange" )) +   labs(y=" AUC (test set) ") +
  ggtitle("Cox-CNN, Case 2, N=1000")

#po_ipcw
setwd( "C:/Users/pabgon/project_2/IPCW_POCNN_so/case2_N1000")
auc1_ipcwpo_so <- npyLoad("auc1_ipcwpo_so.npy")
auc2_ipcwpo_so <- npyLoad("auc2_ipcwpo_so.npy")
auc3_ipcwpo_so <- npyLoad("auc3_ipcwpo_so.npy")
auc4_ipcwpo_so <- npyLoad("auc4_ipcwpo_so.npy")

auc_metric = data.frame(cbind(apply(auc1_ipcwpo_so,1,mean),apply(auc2_ipcwpo_so,1,mean),apply(auc3_ipcwpo_so,1,mean),apply(auc4_ipcwpo_so,1,mean),seq(1,25)))
colnames(auc_metric)<-c("auc1","auc2","auc3","auc4","epoch")


ggplot()+
  geom_line(data=auc_metric,aes(y=auc1,x= epoch,colour="time 1"),size=1 )+
  geom_line(data=auc_metric,aes(y=auc2,x= epoch,colour="time 2"),size=1) +
  geom_line(data=auc_metric,aes(y=auc3,x= epoch,colour="time 3"),size=1) +
  geom_line(data=auc_metric,aes(y=auc4,x= epoch,colour="time 4"),size=1) +
  scale_color_manual(name = "", values = c("time 1" = "blue", "time 2" = "red","time 3" = "green", "time 4" = "orange" )) +   labs(y=" AUC (test set) ") +
  ggtitle("IPCW-POCNN single output, Case 2, N=1000")

#po_mtl_ipcw
setwd( "C:/Users/pabgon/project_2/IPCW_POCNN_mo/case2_N1000")
auc1_ipcwpo_mo <- npyLoad("auc1_ipcwpo_mlt.npy")
auc2_ipcwpo_mo <- npyLoad("auc2_ipcwpo_mlt.npy")
auc3_ipcwpo_mo <- npyLoad("auc3_ipcwpo_mlt.npy")
auc4_ipcwpo_mo <- npyLoad("auc4_ipcwpo_mlt.npy")

auc_metric = data.frame(cbind(apply(auc1_ipcwpo_mo,1,mean),apply(auc2_ipcwpo_mo,1,mean),apply(auc3_ipcwpo_mo,1,mean),apply(auc4_ipcwpo_mo,1,mean),seq(1,25)))
colnames(auc_metric)<-c("auc1","auc2","auc3","auc4","epoch")


ggplot()+
  geom_line(data=auc_metric,aes(y=auc1,x= epoch,colour="time 1"),size=1 )+
  geom_line(data=auc_metric,aes(y=auc2,x= epoch,colour="time 2"),size=1) +
  geom_line(data=auc_metric,aes(y=auc3,x= epoch,colour="time 3"),size=1) +
  geom_line(data=auc_metric,aes(y=auc4,x= epoch,colour="time 4"),size=1) +
  scale_color_manual(name = "", values = c("time 1" = "blue", "time 2" = "red","time 3" = "green", "time 4" = "orange" )) +   labs(y=" AUC (test set) ") +
  ggtitle("IPCW-POCNN multi-output, Case 2, N=1000")


case2_auc1 = data.frame(cbind(auc1_cox[25,],auc1_po_so[25,], auc1_po_mo[25,],auc1_ipcwpo_so[25,],auc1_ipcwpo_mo[25,]))
#labels = c("coxCNN","poCNN1","poCNN2","ipcw_poCNN1","ipcw_poCNN2")
labels = c("cox","po1","po2","po3","po4")
colnames(case2_auc1)<-labels
case2_auc1$id <- seq_len(nrow(case2_auc1))
case2_time1 <- reshape2::melt(case2_auc1, id.vars = c('id'))
# rename
names(case2_time1) <- c('id', 'method', 'auc')
case2_time1$time = 'time 1'

case2_auc2 = data.frame(cbind(auc2_cox[25,],auc2_po_so[25,], auc2_po_mo[25,],auc2_ipcwpo_so[25,],auc2_ipcwpo_mo[25,]))
#labels = c("coxCNN","poCNN")
colnames(case2_auc2)<-labels
case2_auc2$id <- seq_len(nrow(case2_auc2))
case2_time2 <- reshape2::melt(case2_auc2, id.vars = c('id'))
# rename
names(case2_time2) <- c('id', 'method', 'auc')
case2_time2$time = 'time 2'

case2_auc3 = data.frame(cbind(auc3_cox[25,],auc3_po_so[25,], auc3_po_mo[25,],auc3_ipcwpo_so[25,],auc3_ipcwpo_mo[25,]))
#labels = c("coxCNN","poCNN")
colnames(case2_auc3)<-labels
case2_auc3$id <- seq_len(nrow(case2_auc3))
case2_time3 <- reshape2::melt(case2_auc3, id.vars = c('id'))
# rename
names(case2_time3) <- c('id', 'method', 'auc')
case2_time3$time = 'time 3'

case2_auc4 = data.frame(cbind(auc4_cox[25,],auc4_po_so[25,], auc4_po_mo[25,],auc4_ipcwpo_so[25,],auc4_ipcwpo_mo[25,]))
#labels = c("coxCNN","poCNN")
colnames(case2_auc4)<-labels
case2_auc4$id <- seq_len(nrow(case2_auc4))
case2_time4 <- reshape2::melt(case2_auc4, id.vars = c('id'))
# rename
names(case2_time4) <- c('id', 'method', 'auc')
case2_time4$time = 'time 4'



#case2 = rbind(case2_time1,case2_time2,case2_time3, case2_time4, case2_time5)
case2 = rbind(case2_time1,case2_time2,case2_time3, case2_time4)

p <- ggplot(case2, aes(factor(method), auc,fill=time))
p +stat_boxplot(geom = "errorbar", width = 0.5) + geom_violin(trim=FALSE, fill='#A4A4A4', color="lightgrey")+geom_boxplot(width=0.1) +facet_wrap(~time)

p <- ggplot(case2, aes(factor(method), auc,fill=time))
p +stat_boxplot(geom = "errorbar", width = 0.5) + geom_violin(trim=FALSE, fill='#A4A4A4', color="lightgrey",width=.7)+geom_boxplot(aes(fill = factor(method)),width=.3) +facet_wrap(~time,ncol = 5)+
  labs(x="",y="AUC")+ theme(legend.position = "none") +
  theme(panel.grid.major = element_blank(),
        axis.title.x = element_text( size=20),
        axis.title.y = element_text( size=20),
        axis.text.x = element_text( 
          size=15),
        axis.text.y = element_text(size=20)) +
  theme(strip.text.x = element_text(size = 15)) +
  theme(legend.position = "none")+ scale_y_continuous(breaks=seq(0,1,.05))

####################################################
#case 3

#po_so
setwd( "C:/Users/pabgon/project_2/POCNN_so/case3_N1000")

auc1_po_so <- npyLoad("auc1_po_so.npy")
auc2_po_so <- npyLoad("auc2_po_so.npy")
auc3_po_so <- npyLoad("auc3_po_so.npy")
auc4_po_so <- npyLoad("auc4_po_so.npy")

auc_metric = data.frame(cbind(apply(auc1_po_so,1,mean),apply(auc2_po_so,1,mean),apply(auc3_po_so,1,mean),apply(auc4_po_so,1,mean),seq(1,25)))
colnames(auc_metric)<-c("auc1","auc2","auc3","auc4","epoch")


ggplot()+
  geom_line(data=auc_metric,aes(y=auc1,x= epoch,colour="time 1"),size=1 )+
  geom_line(data=auc_metric,aes(y=auc2,x= epoch,colour="time 2"),size=1) +
  geom_line(data=auc_metric,aes(y=auc3,x= epoch,colour="time 3"),size=1) +
  geom_line(data=auc_metric,aes(y=auc4,x= epoch,colour="time 4"),size=1) +
  scale_color_manual(name = "", values = c("time 1" = "blue", "time 2" = "red","time 3" = "green", "time 4" = "orange" )) +   labs(y=" AUC (test set) ") +
  ggtitle("POCNN single output, Case 3, N=1000")

#po_mo
setwd( "C:/Users/pabgon/project_2/POCNN_mo/case3_N1000")
auc1_po_mo <- npyLoad("auc1_po_mlt.npy")
auc2_po_mo <- npyLoad("auc2_po_mlt.npy")
auc3_po_mo <- npyLoad("auc3_po_mlt.npy")
auc4_po_mo <- npyLoad("auc4_po_mlt.npy")

auc_metric = data.frame(cbind(apply(auc1_po_mo,1,mean),apply(auc2_po_mo,1,mean),apply(auc3_po_mo,1,mean),apply(auc4_po_mo,1,mean),seq(1,25)))
colnames(auc_metric)<-c("auc1","auc2","auc3","auc4","epoch")


ggplot()+
  geom_line(data=auc_metric,aes(y=auc1,x= epoch,colour="time 1"),size=1 )+
  geom_line(data=auc_metric,aes(y=auc2,x= epoch,colour="time 2"),size=1) +
  geom_line(data=auc_metric,aes(y=auc3,x= epoch,colour="time 3"),size=1) +
  geom_line(data=auc_metric,aes(y=auc4,x= epoch,colour="time 4"),size=1) +
  scale_color_manual(name = "", values = c("time 1" = "blue", "time 2" = "red","time 3" = "green", "time 4" = "orange" )) +   labs(y=" AUC (test set) ") +
  ggtitle("POCNN multi-output, Case 3, N=1000")

#cox
setwd( "C:/Users/pabgon/project_2/COX_CNN/case3_N1000")

auc1_cox <- npyLoad(file = "auc1_cox.npy")
auc2_cox <- npyLoad(file = "auc2_cox.npy")
auc3_cox <- npyLoad(file = "auc3_cox.npy")
auc4_cox <- npyLoad(file = "auc4_cox.npy")

auc_metric = data.frame(cbind(apply(auc1_cox,1,mean),apply(auc2_cox,1,mean),apply(auc3_cox,1,mean),apply(auc4_cox,1,mean),seq(1,25)))
colnames(auc_metric)<-c("auc1","auc2","auc3","auc4","epoch")


ggplot()+
  geom_line(data=auc_metric,aes(y=auc1,x= epoch,colour="time 1"),size=1 )+
  geom_line(data=auc_metric,aes(y=auc2,x= epoch,colour="time 2"),size=1) +
  geom_line(data=auc_metric,aes(y=auc3,x= epoch,colour="time 3"),size=1) +
  geom_line(data=auc_metric,aes(y=auc4,x= epoch,colour="time 4"),size=1) +
  scale_color_manual(name = "", values = c("time 1" = "blue", "time 2" = "red","time 3" = "green", "time 4" = "orange" )) +   labs(y=" AUC (test set) ") +
  ggtitle("Cox-CNN, Case 3, N=1000")

#po_ipcw
setwd( "C:/Users/pabgon/project_2/IPCW_POCNN_so/case3_N1000")
auc1_ipcwpo_so <- npyLoad("auc1_ipcwpo_so.npy")
auc2_ipcwpo_so <- npyLoad("auc2_ipcwpo_so.npy")
auc3_ipcwpo_so <- npyLoad("auc3_ipcwpo_so.npy")
auc4_ipcwpo_so <- npyLoad("auc4_ipcwpo_so.npy")

auc_metric = data.frame(cbind(apply(auc1_ipcwpo_so,1,mean),apply(auc2_ipcwpo_so,1,mean),apply(auc3_ipcwpo_so,1,mean),apply(auc4_ipcwpo_so,1,mean),seq(1,25)))
colnames(auc_metric)<-c("auc1","auc2","auc3","auc4","epoch")


ggplot()+
  geom_line(data=auc_metric,aes(y=auc1,x= epoch,colour="time 1"),size=1 )+
  geom_line(data=auc_metric,aes(y=auc2,x= epoch,colour="time 2"),size=1) +
  geom_line(data=auc_metric,aes(y=auc3,x= epoch,colour="time 3"),size=1) +
  geom_line(data=auc_metric,aes(y=auc4,x= epoch,colour="time 4"),size=1) +
  scale_color_manual(name = "", values = c("time 1" = "blue", "time 2" = "red","time 3" = "green", "time 4" = "orange" )) +   labs(y=" AUC (test set) ") +
  ggtitle("IPCW-POCNN single output, Case 3, N=1000")

#po_mtl_ipcw
setwd( "C:/Users/pabgon/project_2/IPCW_POCNN_mo/case3_N1000")
auc1_ipcwpo_mo <- npyLoad("auc1_ipcwpo_mlt.npy")
auc2_ipcwpo_mo <- npyLoad("auc2_ipcwpo_mlt.npy")
auc3_ipcwpo_mo <- npyLoad("auc3_ipcwpo_mlt.npy")
auc4_ipcwpo_mo <- npyLoad("auc4_ipcwpo_mlt.npy")

auc_metric = data.frame(cbind(apply(auc1_ipcwpo_mo,1,mean),apply(auc2_ipcwpo_mo,1,mean),apply(auc3_ipcwpo_mo,1,mean),apply(auc4_ipcwpo_mo,1,mean),seq(1,25)))
colnames(auc_metric)<-c("auc1","auc2","auc3","auc4","epoch")


ggplot()+
  geom_line(data=auc_metric,aes(y=auc1,x= epoch,colour="time 1"),size=1 )+
  geom_line(data=auc_metric,aes(y=auc2,x= epoch,colour="time 2"),size=1) +
  geom_line(data=auc_metric,aes(y=auc3,x= epoch,colour="time 3"),size=1) +
  geom_line(data=auc_metric,aes(y=auc4,x= epoch,colour="time 4"),size=1) +
  scale_color_manual(name = "", values = c("time 1" = "blue", "time 2" = "red","time 3" = "green", "time 4" = "orange" )) +   labs(y=" AUC (test set) ") +
  ggtitle("IPCW-POCNN multi-output, Case 3, N=1000")


case3_auc1 = data.frame(cbind(auc1_cox[25,],auc1_po_so[25,], auc1_po_mo[25,],auc1_ipcwpo_so[25,],auc1_ipcwpo_mo[25,]))
#labels = c("coxCNN","poCNN1","poCNN2","ipcw_poCNN1","ipcw_poCNN2")
labels = c("cox","po1","po2","po3","po4")
colnames(case3_auc1)<-labels
case3_auc1$id <- seq_len(nrow(case3_auc1))
case3_time1 <- reshape2::melt(case3_auc1, id.vars = c('id'))
# rename
names(case3_time1) <- c('id', 'method', 'auc')
case3_time1$time = 'time 1'

case3_auc2 = data.frame(cbind(auc2_cox[25,],auc2_po_so[25,], auc2_po_mo[25,],auc2_ipcwpo_so[25,],auc2_ipcwpo_mo[25,]))
#labels = c("coxCNN","poCNN")
colnames(case3_auc2)<-labels
case3_auc2$id <- seq_len(nrow(case3_auc2))
case3_time2 <- reshape2::melt(case3_auc2, id.vars = c('id'))
# rename
names(case3_time2) <- c('id', 'method', 'auc')
case3_time2$time = 'time 2'

case3_auc3 = data.frame(cbind(auc3_cox[25,],auc3_po_so[25,], auc3_po_mo[25,],auc3_ipcwpo_so[25,],auc3_ipcwpo_mo[25,]))
#labels = c("coxCNN","poCNN")
colnames(case3_auc3)<-labels
case3_auc3$id <- seq_len(nrow(case3_auc3))
case3_time3 <- reshape2::melt(case3_auc3, id.vars = c('id'))
# rename
names(case3_time3) <- c('id', 'method', 'auc')
case3_time3$time = 'time 3'

case3_auc4 = data.frame(cbind(auc4_cox[25,],auc4_po_so[25,], auc4_po_mo[25,],auc4_ipcwpo_so[25,],auc4_ipcwpo_mo[25,]))
#labels = c("coxCNN","poCNN")
colnames(case3_auc4)<-labels
case3_auc4$id <- seq_len(nrow(case3_auc4))
case3_time4 <- reshape2::melt(case3_auc4, id.vars = c('id'))
# rename
names(case3_time4) <- c('id', 'method', 'auc')
case3_time4$time = 'time 4'


#case3 = rbind(case3_time1,case3_time2,case3_time3, case3_time4, case3_time5)
case3 = rbind(case3_time1,case3_time2,case3_time3, case3_time4)

p <- ggplot(case3, aes(factor(method), auc,fill=time))
p +stat_boxplot(geom = "errorbar", width = 0.5) + geom_violin(trim=FALSE, fill='#A4A4A4', color="lightgrey")+geom_boxplot(width=0.1) +facet_wrap(~time)

p <- ggplot(case3, aes(factor(method), auc,fill=time))
p +stat_boxplot(geom = "errorbar", width = 0.5) + geom_violin(trim=FALSE, fill='#A4A4A4', color="lightgrey",width=.7)+geom_boxplot(aes(fill = factor(method)),width=.3) +facet_wrap(~time,ncol = 5)+
  labs(x="",y="AUC")+ theme(legend.position = "none") +
  theme(panel.grid.major = element_blank(),
        axis.title.x = element_text( size=20),
        axis.title.y = element_text( size=20),
        axis.text.x = element_text( 
          size=15),
        axis.text.y = element_text(size=20)) +
  theme(strip.text.x = element_text(size = 15)) +
  theme(legend.position = "none")+ scale_y_continuous(breaks=seq(0,1,.05))

###########################################################
#case 4

#po_so
setwd( "C:/Users/pabgon/project_2/POCNN_so/case4_N1000")

auc1_po_so <- npyLoad("auc1_po_so.npy")
auc2_po_so <- npyLoad("auc2_po_so.npy")
auc3_po_so <- npyLoad("auc3_po_so.npy")
auc4_po_so <- npyLoad("auc4_po_so.npy")

auc_metric = data.frame(cbind(apply(auc1_po_so,1,mean),apply(auc2_po_so,1,mean),apply(auc3_po_so,1,mean),apply(auc4_po_so,1,mean),seq(1,25)))
colnames(auc_metric)<-c("auc1","auc2","auc3","auc4","epoch")


ggplot()+
  geom_line(data=auc_metric,aes(y=auc1,x= epoch,colour="time 1"),size=1 )+
  geom_line(data=auc_metric,aes(y=auc2,x= epoch,colour="time 2"),size=1) +
  geom_line(data=auc_metric,aes(y=auc3,x= epoch,colour="time 3"),size=1) +
  geom_line(data=auc_metric,aes(y=auc4,x= epoch,colour="time 4"),size=1) +
  scale_color_manual(name = "", values = c("time 1" = "blue", "time 2" = "red","time 3" = "green", "time 4" = "orange" )) +   labs(y=" AUC (test set) ") +
  ggtitle("POCNN single output, Case 4, N=1000")

#po_mo
setwd( "C:/Users/pabgon/project_2/POCNN_mo/case4_N1000")
auc1_po_mo <- npyLoad("auc1_po_mlt.npy")
auc2_po_mo <- npyLoad("auc2_po_mlt.npy")
auc3_po_mo <- npyLoad("auc3_po_mlt.npy")
auc4_po_mo <- npyLoad("auc4_po_mlt.npy")

auc_metric = data.frame(cbind(apply(auc1_po_mo,1,mean),apply(auc2_po_mo,1,mean),apply(auc3_po_mo,1,mean),apply(auc4_po_mo,1,mean),seq(1,25)))
colnames(auc_metric)<-c("auc1","auc2","auc3","auc4","epoch")


ggplot()+
  geom_line(data=auc_metric,aes(y=auc1,x= epoch,colour="time 1"),size=1 )+
  geom_line(data=auc_metric,aes(y=auc2,x= epoch,colour="time 2"),size=1) +
  geom_line(data=auc_metric,aes(y=auc3,x= epoch,colour="time 3"),size=1) +
  geom_line(data=auc_metric,aes(y=auc4,x= epoch,colour="time 4"),size=1) +
  scale_color_manual(name = "", values = c("time 1" = "blue", "time 2" = "red","time 3" = "green", "time 4" = "orange" )) +   labs(y=" AUC (test set) ") +
  ggtitle("POCNN multi-output, Case 4, N=1000")

#cox
setwd( "C:/Users/pabgon/project_2/COX_CNN/case4_N1000")

auc1_cox <- npyLoad(file = "auc1_cox.npy")
auc2_cox <- npyLoad(file = "auc2_cox.npy")
auc3_cox <- npyLoad(file = "auc3_cox.npy")
auc4_cox <- npyLoad(file = "auc4_cox.npy")

auc_metric = data.frame(cbind(apply(auc1_cox,1,mean),apply(auc2_cox,1,mean),apply(auc3_cox,1,mean),apply(auc4_cox,1,mean),seq(1,25)))
colnames(auc_metric)<-c("auc1","auc2","auc3","auc4","epoch")


ggplot()+
  geom_line(data=auc_metric,aes(y=auc1,x= epoch,colour="time 1"),size=1 )+
  geom_line(data=auc_metric,aes(y=auc2,x= epoch,colour="time 2"),size=1) +
  geom_line(data=auc_metric,aes(y=auc3,x= epoch,colour="time 3"),size=1) +
  geom_line(data=auc_metric,aes(y=auc4,x= epoch,colour="time 4"),size=1) +
  scale_color_manual(name = "", values = c("time 1" = "blue", "time 2" = "red","time 3" = "green", "time 4" = "orange" )) +   labs(y=" AUC (test set) ") +
  ggtitle("Cox-CNN, Case 4, N=1000")

#po_ipcw
setwd( "C:/Users/pabgon/project_2/IPCW_POCNN_so/case4_N1000")
auc1_ipcwpo_so <- npyLoad("auc1_ipcwpo_so.npy")
auc2_ipcwpo_so <- npyLoad("auc2_ipcwpo_so.npy")
auc3_ipcwpo_so <- npyLoad("auc3_ipcwpo_so.npy")
auc4_ipcwpo_so <- npyLoad("auc4_ipcwpo_so.npy")


auc_metric = data.frame(cbind(apply(auc1_ipcwpo_so,1,mean),apply(auc2_ipcwpo_so,1,mean),apply(auc3_ipcwpo_so,1,mean),apply(auc4_ipcwpo_so,1,mean),seq(1,25)))
colnames(auc_metric)<-c("auc1","auc2","auc3","auc4","epoch")


ggplot()+
  geom_line(data=auc_metric,aes(y=auc1,x= epoch,colour="time 1"),size=1 )+
  geom_line(data=auc_metric,aes(y=auc2,x= epoch,colour="time 2"),size=1) +
  geom_line(data=auc_metric,aes(y=auc3,x= epoch,colour="time 3"),size=1) +
  geom_line(data=auc_metric,aes(y=auc4,x= epoch,colour="time 4"),size=1) +
  scale_color_manual(name = "", values = c("time 1" = "blue", "time 2" = "red","time 3" = "green", "time 4" = "orange" )) +   labs(y=" AUC (test set) ") +
  ggtitle("IPCW-POCNN single output, Case 4, N=1000")


#po_mtl_ipcw
setwd( "C:/Users/pabgon/project_2/IPCW_POCNN_mo/case4_N1000")
auc1_ipcwpo_mo <- npyLoad("auc1_ipcwpo_mlt.npy")
auc2_ipcwpo_mo <- npyLoad("auc2_ipcwpo_mlt.npy")
auc3_ipcwpo_mo <- npyLoad("auc3_ipcwpo_mlt.npy")
auc4_ipcwpo_mo <- npyLoad("auc4_ipcwpo_mlt.npy")

auc_metric = data.frame(cbind(apply(auc1_ipcwpo_mo,1,mean),apply(auc2_ipcwpo_mo,1,mean),apply(auc3_ipcwpo_mo,1,mean),apply(auc4_ipcwpo_mo,1,mean),seq(1,25)))
colnames(auc_metric)<-c("auc1","auc2","auc3","auc4","epoch")


ggplot()+
  geom_line(data=auc_metric,aes(y=auc1,x= epoch,colour="time 1"),size=1 )+
  geom_line(data=auc_metric,aes(y=auc2,x= epoch,colour="time 2"),size=1) +
  geom_line(data=auc_metric,aes(y=auc3,x= epoch,colour="time 3"),size=1) +
  geom_line(data=auc_metric,aes(y=auc4,x= epoch,colour="time 4"),size=1) +
  scale_color_manual(name = "", values = c("time 1" = "blue", "time 2" = "red","time 3" = "green", "time 4" = "orange" )) +   labs(y=" AUC (test set) ") +
  ggtitle("IPCW-POCNN multi-output, Case 4, N=1000")


case4_auc1 = data.frame(cbind(auc1_cox[25,],auc1_po_so[25,], auc1_po_mo[25,],auc1_ipcwpo_so[25,],auc1_ipcwpo_mo[25,]))
#labels = c("coxCNN","poCNN1","poCNN2","ipcw_poCNN1","ipcw_poCNN2")
labels = c("cox","po1","po2","po3","po4")
colnames(case4_auc1)<-labels
case4_auc1$id <- seq_len(nrow(case4_auc1))
case4_time1 <- reshape2::melt(case4_auc1, id.vars = c('id'))
# rename
names(case4_time1) <- c('id', 'method', 'auc')
case4_time1$time = 'time 1'

case4_auc2 = data.frame(cbind(auc2_cox[25,],auc2_po_so[25,], auc2_po_mo[25,],auc2_ipcwpo_so[25,],auc2_ipcwpo_mo[25,]))
#labels = c("coxCNN","poCNN")
colnames(case4_auc2)<-labels
case4_auc2$id <- seq_len(nrow(case4_auc2))
case4_time2 <- reshape2::melt(case4_auc2, id.vars = c('id'))
# rename
names(case4_time2) <- c('id', 'method', 'auc')
case4_time2$time = 'time 2'

case4_auc3 = data.frame(cbind(auc3_cox[25,],auc3_po_so[25,], auc3_po_mo[25,],auc3_ipcwpo_so[25,],auc3_ipcwpo_mo[25,]))
#labels = c("coxCNN","poCNN")
colnames(case4_auc3)<-labels
case4_auc3$id <- seq_len(nrow(case4_auc3))
case4_time3 <- reshape2::melt(case4_auc3, id.vars = c('id'))
# rename
names(case4_time3) <- c('id', 'method', 'auc')
case4_time3$time = 'time 3'

case4_auc4 = data.frame(cbind(auc4_cox[25,],auc4_po_so[25,], auc4_po_mo[25,],auc4_ipcwpo_so[25,],auc4_ipcwpo_mo[25,]))
#labels = c("coxCNN","poCNN")
colnames(case4_auc4)<-labels
case4_auc4$id <- seq_len(nrow(case4_auc4))
case4_time4 <- reshape2::melt(case4_auc4, id.vars = c('id'))
# rename
names(case4_time4) <- c('id', 'method', 'auc')
case4_time4$time = 'time 4'




#case4 = rbind(case4_time1,case4_time2,case4_time3, case4_time4, case4_time5)
case4 = rbind(case4_time1,case4_time2,case4_time3, case4_time4)

p <- ggplot(case4, aes(factor(method), auc,fill=time))
p +stat_boxplot(geom = "errorbar", width = 0.5) + geom_violin(trim=FALSE, fill='#A4A4A4', color="lightgrey")+geom_boxplot(width=0.1) +facet_wrap(~time)

p <- ggplot(case4, aes(factor(method), auc,fill=time))
p +stat_boxplot(geom = "errorbar", width = 0.5) + geom_violin(trim=FALSE, fill='#A4A4A4', color="lightgrey",width=.7)+geom_boxplot(aes(fill = factor(method)),width=.3) +facet_wrap(~time,ncol = 5)+
  labs(x="",y="AUC")+ theme(legend.position = "none") +
  theme(panel.grid.major = element_blank(),
        axis.title.x = element_text( size=20),
        axis.title.y = element_text( size=20),
        axis.text.x = element_text( 
          size=15),
        axis.text.y = element_text(size=20)) +
  theme(strip.text.x = element_text(size = 15)) +
  theme(legend.position = "none")+ scale_y_continuous(breaks=seq(0,1,.05))
###########################################################
#case 5

#po_so
setwd( "C:/Users/pabgon/project_2/POCNN_so/case5_N1000")

auc1_po_so <- npyLoad("auc1_po_so.npy")
auc2_po_so <- npyLoad("auc2_po_so.npy")
auc3_po_so <- npyLoad("auc3_po_so.npy")
auc4_po_so <- npyLoad("auc4_po_so.npy")

auc_metric = data.frame(cbind(apply(auc1_po_so,1,mean),apply(auc2_po_so,1,mean),apply(auc3_po_so,1,mean),apply(auc4_po_so,1,mean),seq(1,25)))
colnames(auc_metric)<-c("auc1","auc2","auc3","auc4","epoch")


ggplot()+
  geom_line(data=auc_metric,aes(y=auc1,x= epoch,colour="time 1"),size=1 )+
  geom_line(data=auc_metric,aes(y=auc2,x= epoch,colour="time 2"),size=1) +
  geom_line(data=auc_metric,aes(y=auc3,x= epoch,colour="time 3"),size=1) +
  geom_line(data=auc_metric,aes(y=auc4,x= epoch,colour="time 4"),size=1) +
  scale_color_manual(name = "", values = c("time 1" = "blue", "time 2" = "red","time 3" = "green", "time 4" = "orange" )) +   labs(y=" AUC (test set) ") +
  ggtitle("POCNN single output, Case 5, N=1000")

#po_mo
setwd( "C:/Users/pabgon/project_2/POCNN_mo/case5_N1000")
auc1_po_mo <- npyLoad("auc1_po_mlt.npy")
auc2_po_mo <- npyLoad("auc2_po_mlt.npy")
auc3_po_mo <- npyLoad("auc3_po_mlt.npy")
auc4_po_mo <- npyLoad("auc4_po_mlt.npy")

auc_metric = data.frame(cbind(apply(auc1_po_mo,1,mean),apply(auc2_po_mo,1,mean),apply(auc3_po_mo,1,mean),apply(auc4_po_mo,1,mean),seq(1,25)))
colnames(auc_metric)<-c("auc1","auc2","auc3","auc4","epoch")


ggplot()+
  geom_line(data=auc_metric,aes(y=auc1,x= epoch,colour="time 1"),size=1 )+
  geom_line(data=auc_metric,aes(y=auc2,x= epoch,colour="time 2"),size=1) +
  geom_line(data=auc_metric,aes(y=auc3,x= epoch,colour="time 3"),size=1) +
  geom_line(data=auc_metric,aes(y=auc4,x= epoch,colour="time 4"),size=1) +
  scale_color_manual(name = "", values = c("time 1" = "blue", "time 2" = "red","time 3" = "green", "time 4" = "orange" )) +   labs(y=" AUC (test set) ") +
  ggtitle("POCNN multi-output, Case 5, N=1000")


#cox
setwd( "C:/Users/pabgon/project_2/COX_CNN/case5_N1000")

auc1_cox <- npyLoad(file = "auc1_cox.npy")
auc2_cox <- npyLoad(file = "auc2_cox.npy")
auc3_cox <- npyLoad(file = "auc3_cox.npy")
auc4_cox <- npyLoad(file = "auc4_cox.npy")

auc_metric = data.frame(cbind(apply(auc1_cox,1,mean),apply(auc2_cox,1,mean),apply(auc3_cox,1,mean),apply(auc4_cox,1,mean),seq(1,25)))
colnames(auc_metric)<-c("auc1","auc2","auc3","auc4","epoch")


ggplot()+
  geom_line(data=auc_metric,aes(y=auc1,x= epoch,colour="time 1"),size=1 )+
  geom_line(data=auc_metric,aes(y=auc2,x= epoch,colour="time 2"),size=1) +
  geom_line(data=auc_metric,aes(y=auc3,x= epoch,colour="time 3"),size=1) +
  geom_line(data=auc_metric,aes(y=auc4,x= epoch,colour="time 4"),size=1) +
  scale_color_manual(name = "", values = c("time 1" = "blue", "time 2" = "red","time 3" = "green", "time 4" = "orange" )) +   labs(y=" AUC (test set) ") +
  ggtitle("Cox-CNN, Case 5, N=1000")

#po_ipcw
setwd( "C:/Users/pabgon/project_2/IPCW_POCNN_so/case5_N1000")
auc1_ipcwpo_so <- npyLoad("auc1_ipcwpo_so.npy")
auc2_ipcwpo_so <- npyLoad("auc2_ipcwpo_so.npy")
auc3_ipcwpo_so <- npyLoad("auc3_ipcwpo_so.npy")
auc4_ipcwpo_so <- npyLoad("auc4_ipcwpo_so.npy")

auc_metric = data.frame(cbind(apply(auc1_ipcwpo_so,1,mean),apply(auc2_ipcwpo_so,1,mean),apply(auc3_ipcwpo_so,1,mean),apply(auc4_ipcwpo_so,1,mean),seq(1,25)))
colnames(auc_metric)<-c("auc1","auc2","auc3","auc4","epoch")


ggplot()+
  geom_line(data=auc_metric,aes(y=auc1,x= epoch,colour="time 1"),size=1 )+
  geom_line(data=auc_metric,aes(y=auc2,x= epoch,colour="time 2"),size=1) +
  geom_line(data=auc_metric,aes(y=auc3,x= epoch,colour="time 3"),size=1) +
  geom_line(data=auc_metric,aes(y=auc4,x= epoch,colour="time 4"),size=1) +
  scale_color_manual(name = "", values = c("time 1" = "blue", "time 2" = "red","time 3" = "green", "time 4" = "orange" )) +   labs(y=" AUC (test set) ") +
  ggtitle("IPCW-POCNN single output, Case 5, N=1000")

#po_mtl_ipcw
setwd( "C:/Users/pabgon/project_2/IPCW_POCNN_mo/case5_N1000")
auc1_ipcwpo_mo <- npyLoad("auc1_ipcwpo_mlt.npy")
auc2_ipcwpo_mo <- npyLoad("auc2_ipcwpo_mlt.npy")
auc3_ipcwpo_mo <- npyLoad("auc3_ipcwpo_mlt.npy")
auc4_ipcwpo_mo <- npyLoad("auc4_ipcwpo_mlt.npy")

auc_metric = data.frame(cbind(apply(auc1_ipcwpo_mo,1,mean),apply(auc2_ipcwpo_mo,1,mean),apply(auc3_ipcwpo_mo,1,mean),apply(auc4_ipcwpo_mo,1,mean),seq(1,25)))
colnames(auc_metric)<-c("auc1","auc2","auc3","auc4","epoch")


ggplot()+
  geom_line(data=auc_metric,aes(y=auc1,x= epoch,colour="time 1"),size=1 )+
  geom_line(data=auc_metric,aes(y=auc2,x= epoch,colour="time 2"),size=1) +
  geom_line(data=auc_metric,aes(y=auc3,x= epoch,colour="time 3"),size=1) +
  geom_line(data=auc_metric,aes(y=auc4,x= epoch,colour="time 4"),size=1) +
  scale_color_manual(name = "", values = c("time 1" = "blue", "time 2" = "red","time 3" = "green", "time 4" = "orange" )) +   labs(y=" AUC (test set) ") +
  ggtitle("IPCW-POCNN multi-output, Case 5, N=1000")



case5_auc1 = data.frame(cbind(auc1_cox[25,],auc1_po_so[25,], auc1_po_mo[25,],auc1_ipcwpo_so[25,],auc1_ipcwpo_mo[25,]))
#labels = c("coxCNN","poCNN1","poCNN2","ipcw_poCNN1","ipcw_poCNN2")
labels = c("cox","po1","po2","po3","po4")
colnames(case5_auc1)<-labels
case5_auc1$id <- seq_len(nrow(case5_auc1))
case5_time1 <- reshape2::melt(case5_auc1, id.vars = c('id'))
# rename
names(case5_time1) <- c('id', 'method', 'auc')
case5_time1$time = 'time 1'

case5_auc2 = data.frame(cbind(auc2_cox[25,],auc2_po_so[25,], auc2_po_mo[25,],auc2_ipcwpo_so[25,],auc2_ipcwpo_mo[25,]))
#labels = c("coxCNN","poCNN")
colnames(case5_auc2)<-labels
case5_auc2$id <- seq_len(nrow(case5_auc2))
case5_time2 <- reshape2::melt(case5_auc2, id.vars = c('id'))
# rename
names(case5_time2) <- c('id', 'method', 'auc')
case5_time2$time = 'time 2'

case5_auc3 = data.frame(cbind(auc3_cox[25,],auc3_po_so[25,], auc3_po_mo[25,],auc3_ipcwpo_so[25,],auc3_ipcwpo_mo[25,]))
#labels = c("coxCNN","poCNN")
colnames(case5_auc3)<-labels
case5_auc3$id <- seq_len(nrow(case5_auc3))
case5_time3 <- reshape2::melt(case5_auc3, id.vars = c('id'))
# rename
names(case5_time3) <- c('id', 'method', 'auc')
case5_time3$time = 'time 3'

case5_auc4 = data.frame(cbind(auc4_cox[25,],auc4_po_so[25,], auc4_po_mo[25,],auc4_ipcwpo_so[25,],auc4_ipcwpo_mo[25,]))
#labels = c("coxCNN","poCNN")
colnames(case5_auc4)<-labels
case5_auc4$id <- seq_len(nrow(case5_auc4))
case5_time4 <- reshape2::melt(case5_auc4, id.vars = c('id'))
# rename
names(case5_time4) <- c('id', 'method', 'auc')
case5_time4$time = 'time 4'




#case5 = rbind(case5_time1,case5_time2,case5_time3, case5_time4, case5_time5)
case5 = rbind(case5_time1,case5_time2,case5_time3, case5_time4)

#p <- ggplot(CASE5, aes(factor(method), auc,fill=time))
#p +stat_boxplot(geom = "errorbar", width = 0.5) + geom_violin(trim=FALSE, fill='#A4A4A4', color="lightgrey")+geom_boxplot(width=0.1) +facet_wrap(~time)

#p <- ggplot(case5, aes(factor(method), auc,fill=time))
#p +stat_boxplot(geom = "errorbar", width = 0.5) + geom_violin(trim=FALSE, fill='#A4A4A4', color="lightgrey",width=.7)+geom_boxplot(aes(fill = factor(method)),width=.3) +facet_wrap(~time,ncol = 5)+
#  labs(x="",y="AUC")+ theme(legend.position = "none") +
#  theme(panel.grid.major = element_blank(),
#        axis.title.x = element_text( size=20),
#        axis.title.y = element_text( size=20),
#        axis.text.x = element_text( 
#          size=15),
#        axis.text.y = element_text(size=20)) +
#  theme(strip.text.x = element_text(size = 15)) +
#  theme(legend.position = "none")+ scale_y_continuous(breaks=seq(0,1,.05))

#case5_auc = data.frame(auc1_pseudo-auc1_cox)
#boxplot(case5_auc)
#labels = c("coxCNN","poCNN")
#colnames(case5_auc1)<-labels
#case5_auc1$id <- seq_len(nrow(case5_auc1))
#case5_time1 <- reshape2::melt(case5_auc1, id.vars = c('id'))
# rename
#names(case5_time1) <- c('id', 'method', 'auc')

#case5_1 = data.frame(auc1_pseudo-auc1_cox)
#colnames(case5_1) = 'AUC'
#case5_1$time = 'time 1'

#case5_1 = data.frame(auc1_pseudo-auc1_cox)
#colnames(case5_1) = 'AUC'
#case5_1$time = 'time 1'


###########################################################
#case 6

#po_so
setwd( "C:/Users/pabgon/project_2/POCNN_so/case6_N1000")

auc1_po_so <- npyLoad("auc1_po_so.npy")
auc2_po_so <- npyLoad("auc2_po_so.npy")
auc3_po_so <- npyLoad("auc3_po_so.npy")
auc4_po_so <- npyLoad("auc4_po_so.npy")

auc_metric = data.frame(cbind(apply(auc1_po_so,1,mean),apply(auc2_po_so,1,mean),apply(auc3_po_so,1,mean),apply(auc4_po_so,1,mean),seq(1,25)))
colnames(auc_metric)<-c("auc1","auc2","auc3","auc4","epoch")


ggplot()+
  geom_line(data=auc_metric,aes(y=auc1,x= epoch,colour="time 1"),size=1 )+
  geom_line(data=auc_metric,aes(y=auc2,x= epoch,colour="time 2"),size=1) +
  geom_line(data=auc_metric,aes(y=auc3,x= epoch,colour="time 3"),size=1) +
  geom_line(data=auc_metric,aes(y=auc4,x= epoch,colour="time 4"),size=1) +
  scale_color_manual(name = "", values = c("time 1" = "blue", "time 2" = "red","time 3" = "green", "time 4" = "orange" )) +   labs(y=" AUC (test set) ") +
  ggtitle("POCNN single output, Case 6, N=1000")

#po_mo
setwd( "C:/Users/pabgon/project_2/POCNN_mo/case6_N1000")
auc1_po_mo <- npyLoad("auc1_po_mlt.npy")
auc2_po_mo <- npyLoad("auc2_po_mlt.npy")
auc3_po_mo <- npyLoad("auc3_po_mlt.npy")
auc4_po_mo <- npyLoad("auc4_po_mlt.npy")

auc_metric = data.frame(cbind(apply(auc1_po_mo,1,mean),apply(auc2_po_mo,1,mean),apply(auc3_po_mo,1,mean),apply(auc4_po_mo,1,mean),seq(1,25)))
colnames(auc_metric)<-c("auc1","auc2","auc3","auc4","epoch")


ggplot()+
  geom_line(data=auc_metric,aes(y=auc1,x= epoch,colour="time 1"),size=1 )+
  geom_line(data=auc_metric,aes(y=auc2,x= epoch,colour="time 2"),size=1) +
  geom_line(data=auc_metric,aes(y=auc3,x= epoch,colour="time 3"),size=1) +
  geom_line(data=auc_metric,aes(y=auc4,x= epoch,colour="time 4"),size=1) +
  scale_color_manual(name = "", values = c("time 1" = "blue", "time 2" = "red","time 3" = "green", "time 4" = "orange" )) +   labs(y=" AUC (test set) ") +
  ggtitle("POCNN multi-output, Case 6, N=1000")

#cox
setwd( "C:/Users/pabgon/project_2/COX_CNN/case6_N1000")

auc1_cox <- npyLoad(file = "auc1_cox.npy")
auc2_cox <- npyLoad(file = "auc2_cox.npy")
auc3_cox <- npyLoad(file = "auc3_cox.npy")
auc4_cox <- npyLoad(file = "auc4_cox.npy")

auc_metric = data.frame(cbind(apply(auc1_cox,1,mean),apply(auc2_cox,1,mean),apply(auc3_cox,1,mean),apply(auc4_cox,1,mean),seq(1,25)))
colnames(auc_metric)<-c("auc1","auc2","auc3","auc4","epoch")


ggplot()+
  geom_line(data=auc_metric,aes(y=auc1,x= epoch,colour="time 1"),size=1 )+
  geom_line(data=auc_metric,aes(y=auc2,x= epoch,colour="time 2"),size=1) +
  geom_line(data=auc_metric,aes(y=auc3,x= epoch,colour="time 3"),size=1) +
  geom_line(data=auc_metric,aes(y=auc4,x= epoch,colour="time 4"),size=1) +
  scale_color_manual(name = "", values = c("time 1" = "blue", "time 2" = "red","time 3" = "green", "time 4" = "orange" )) +   labs(y=" AUC (test set) ") +
  ggtitle("Cox-CNN, Case 6, N=1000")

#po_ipcw
setwd( "C:/Users/pabgon/project_2/IPCW_POCNN_so/case6_N1000")
auc1_ipcwpo_so <- npyLoad("auc1_ipcwpo_so.npy")
auc2_ipcwpo_so <- npyLoad("auc2_ipcwpo_so.npy")
auc3_ipcwpo_so <- npyLoad("auc3_ipcwpo_so.npy")
auc4_ipcwpo_so <- npyLoad("auc4_ipcwpo_so.npy")

auc_metric = data.frame(cbind(apply(auc1_ipcwpo_so,1,mean),apply(auc2_ipcwpo_so,1,mean),apply(auc3_ipcwpo_so,1,mean),apply(auc4_ipcwpo_so,1,mean),seq(1,25)))
colnames(auc_metric)<-c("auc1","auc2","auc3","auc4","epoch")


ggplot()+
  geom_line(data=auc_metric,aes(y=auc1,x= epoch,colour="time 1"),size=1 )+
  geom_line(data=auc_metric,aes(y=auc2,x= epoch,colour="time 2"),size=1) +
  geom_line(data=auc_metric,aes(y=auc3,x= epoch,colour="time 3"),size=1) +
  geom_line(data=auc_metric,aes(y=auc4,x= epoch,colour="time 4"),size=1) +
  scale_color_manual(name = "", values = c("time 1" = "blue", "time 2" = "red","time 3" = "green", "time 4" = "orange" )) +   labs(y=" AUC (test set) ") +
  ggtitle("IPCW-POCNN single output, Case 6, N=1000")

#po_mtl_ipcw
setwd( "C:/Users/pabgon/project_2/IPCW_POCNN_mo/case6_N1000")
auc1_ipcwpo_mo <- npyLoad("auc1_ipcwpo_mlt.npy")
auc2_ipcwpo_mo <- npyLoad("auc2_ipcwpo_mlt.npy")
auc3_ipcwpo_mo <- npyLoad("auc3_ipcwpo_mlt.npy")
auc4_ipcwpo_mo <- npyLoad("auc4_ipcwpo_mlt.npy")

auc_metric = data.frame(cbind(apply(auc1_ipcwpo_mo,1,mean),apply(auc2_ipcwpo_mo,1,mean),apply(auc3_ipcwpo_mo,1,mean),apply(auc4_ipcwpo_mo,1,mean),seq(1,25)))
colnames(auc_metric)<-c("auc1","auc2","auc3","auc4","epoch")


ggplot()+
  geom_line(data=auc_metric,aes(y=auc1,x= epoch,colour="time 1"),size=1 )+
  geom_line(data=auc_metric,aes(y=auc2,x= epoch,colour="time 2"),size=1) +
  geom_line(data=auc_metric,aes(y=auc3,x= epoch,colour="time 3"),size=1) +
  geom_line(data=auc_metric,aes(y=auc4,x= epoch,colour="time 4"),size=1) +
  scale_color_manual(name = "", values = c("time 1" = "blue", "time 2" = "red","time 3" = "green", "time 4" = "orange" )) +   labs(y=" AUC (test set) ") +
  ggtitle("IPCW-POCNN multi-output, Case 6, N=1000")


case6_auc1 = data.frame(cbind(auc1_cox[25,],auc1_po_so[25,], auc1_po_mo[25,],auc1_ipcwpo_so[25,],auc1_ipcwpo_mo[25,]))
#labels = c("coxCNN","poCNN1","poCNN2","ipcw_poCNN1","ipcw_poCNN2")
labels = c("cox","po1","po2","po3","po4")
colnames(case6_auc1)<-labels
case6_auc1$id <- seq_len(nrow(case6_auc1))
case6_time1 <- reshape2::melt(case6_auc1, id.vars = c('id'))
# rename
names(case6_time1) <- c('id', 'method', 'auc')
case6_time1$time = 'time 1'

case6_auc2 = data.frame(cbind(auc2_cox[25,],auc2_po_so[25,], auc2_po_mo[25,],auc2_ipcwpo_so[25,],auc2_ipcwpo_mo[25,]))
#labels = c("coxCNN","poCNN")
colnames(case6_auc2)<-labels
case6_auc2$id <- seq_len(nrow(case6_auc2))
case6_time2 <- reshape2::melt(case6_auc2, id.vars = c('id'))
# rename
names(case6_time2) <- c('id', 'method', 'auc')
case6_time2$time = 'time 2'

case6_auc3 = data.frame(cbind(auc3_cox[25,],auc3_po_so[25,], auc3_po_mo[25,],auc3_ipcwpo_so[25,],auc3_ipcwpo_mo[25,]))
#labels = c("coxCNN","poCNN")
colnames(case6_auc3)<-labels
case6_auc3$id <- seq_len(nrow(case6_auc3))
case6_time3 <- reshape2::melt(case6_auc3, id.vars = c('id'))
# rename
names(case6_time3) <- c('id', 'method', 'auc')
case6_time3$time = 'time 3'

case6_auc4 = data.frame(cbind(auc4_cox[25,],auc4_po_so[25,], auc4_po_mo[25,],auc4_ipcwpo_so[25,],auc4_ipcwpo_mo[25,]))
#labels = c("coxCNN","poCNN")
colnames(case6_auc4)<-labels
case6_auc4$id <- seq_len(nrow(case6_auc4))
case6_time4 <- reshape2::melt(case6_auc4, id.vars = c('id'))
# rename
names(case6_time4) <- c('id', 'method', 'auc')
case6_time4$time = 'time 4'


case6 = rbind(case6_time1,case6_time2,case6_time3, case6_time4)
#case6 = rbind(case6_time1,case6_time2,case6_time3, case6_time4)

p <- ggplot(case6, aes(factor(method), auc,fill=time))
p +stat_boxplot(geom = "errorbar", width = 0.5) + geom_violin(trim=FALSE, fill='#A4A4A4', color="lightgrey")+geom_boxplot(width=0.1) +facet_wrap(~time)

p <- ggplot(case6, aes(factor(method), auc,fill=time))
p +stat_boxplot(geom = "errorbar", width = 0.5) + geom_violin(trim=FALSE, fill='#A4A4A4', color="lightgrey",width=.7)+geom_boxplot(aes(fill = factor(method)),width=.3) +facet_wrap(~time,ncol = 5)+
  labs(x="",y="AUC")+ theme(legend.position = "none") +
  theme(panel.grid.major = element_blank(),
        axis.title.x = element_text( size=20),
        axis.title.y = element_text( size=20),
        axis.text.x = element_text( 
          size=15),
        axis.text.y = element_text(size=20)) +
  theme(strip.text.x = element_text(size = 15)) +
  theme(legend.position = "none")+ scale_y_continuous(breaks=seq(0,1,.05))

boxplot(case1_auc1,ylab="AUC")
######################################

case1$scenario = 'Case 1'
case2$scenario = 'Case 2'
case3$scenario = 'Case 3'
case4$scenario = 'Case 4'
case5$scenario = 'Case 5'
case6$scenario = 'Case 6'

all_cases = rbind(case1, case2, case3, case4, case5, case6)
all_cases_1_2_3 = rbind(case1, case2, case3)
all_cases_4_5_6 = rbind(case4, case5, case6)


my_comparisons <- list( c("cox", "po1"), c("cox", "po2"), c("cox", "po3"),c("cox", "po4") )

p = ggplot(all_cases_1_2_3, aes(factor(method), auc,fill=method))
p +  geom_boxplot()+
  theme_bw()+
  facet_grid(scenario~time,scales='free_x', space='free_x') + scale_y_continuous(limits=c(0,1.1)) + 
  stat_compare_means(comparisons = my_comparisons, method = "t.test", label = "p.signif", size = 4) +
  labs(x="",y="AUC")+ theme(legend.position = "none")+
  theme(panel.grid.major = element_blank(),
        axis.title.x = element_text( size=15),
        axis.title.y = element_text( size=15),
        axis.text.x = element_text( 
          size=15),
        axis.text.y = element_text(size=15)) +
  theme(strip.text.x = element_text(size = 15)) + 
  theme(strip.text.y = element_text(size = 15)) + theme_bw(base_size = 15)+  theme(legend.position = "none")


p = ggplot(all_cases_4_5_6, aes(factor(method), auc,fill=method))
p +  geom_boxplot()+
  theme_bw()+
  facet_grid(scenario~time,scales='free_x', space='free_x') + scale_y_continuous(limits=c(0.2,1.1)) + 
  stat_compare_means(comparisons = my_comparisons, method = "t.test", label = "p.signif", size = 4) +
  labs(x="",y="AUC")+ theme(legend.position = "none")+
  theme(panel.grid.major = element_blank(),
        axis.title.x = element_text( size=15),
        axis.title.y = element_text( size=15),
        axis.text.x = element_text( 
          size=15),
        axis.text.y = element_text(size=15)) +
  theme(strip.text.x = element_text(size = 15)) + 
  theme(strip.text.y = element_text(size = 15)) + theme_bw(base_size = 15)+  theme(legend.position = "none")

##### this is the plot!!!!
p = ggplot(all_cases, aes(factor(method), auc,fill=method))
p +  geom_boxplot()+
  theme_bw()+
  facet_grid(scenario~time,scales = "free", space = "free") + scale_y_continuous(limits=c(0.2,1.1)) + 
  stat_compare_means(comparisons = my_comparisons, method = "t.test",label = "p.signif",label.y = c(.8,.85,.9,.95) ,vjust = .5, step.increase = 0.05) +
  labs(x="",y="AUC")+ theme(legend.position = "none")+
  theme(panel.grid.major = element_blank(),
        axis.title.x = element_text( size=15),
        axis.title.y = element_text( size=15),
        axis.text.x = element_text( 
          size=15),
        axis.text.y = element_text(size=15)) +
  theme(strip.text.x = element_text(size = 15)) + 
  theme(strip.text.y = element_text(size = 15)) + theme_bw(base_size = 15)+  theme(legend.position = "none")


p = ggplot(all_cases, aes(factor(method), auc,fill=method))
p +  geom_boxplot()+
  theme_bw()+
  facet_grid(scenario~time,scales = "free", space = "free") + scale_y_continuous(limits=c(0.2,1.1)) + 
  stat_compare_means(comparisons = my_comparisons, method = "wilcox.test",label = "p.signif",label.y = c(.8,.85,.9,.95) ,vjust = .5, step.increase = 0.05) +
  labs(x="",y="AUC")+ theme(legend.position = "none")+
  theme(panel.grid.major = element_blank(),
        axis.title.x = element_text( size=15),
        axis.title.y = element_text( size=15),
        axis.text.x = element_text( 
          size=15),
        axis.text.y = element_text(size=15)) +
  theme(strip.text.x = element_text(size = 15)) + 
  theme(strip.text.y = element_text(size = 15)) + theme_bw(base_size = 15)+  theme(legend.position = "none")
########################################################



p + stat_boxplot(geom = "errorbar", width = 0.5) + geom_violin(trim=FALSE, fill='#A4A4A4', color="lightgrey",width=.7) + geom_boxplot(width=.3)+
  theme_bw()+
  facet_grid(scenario~time) + labs(x="",y="AUC")+ theme(legend.position = "none")+
  theme(panel.grid.major = element_blank(),
        axis.title.x = element_text( size=20),
        axis.title.y = element_text( size=20),
        axis.text.x = element_text( 
          size=15),
        axis.text.y = element_text(size=20)) +
  theme(strip.text.x = element_text(size = 15)) + 
  theme(strip.text.y = element_text(size = 15)) + theme_bw(base_size = 18)+  theme(legend.position = "none")


p = ggplot(all_cases_4_5_6, aes(factor(method), auc,fill=method))
p + stat_boxplot(geom = "errorbar", width = 0.5) + geom_violin(trim=FALSE, fill='#A4A4A4', color="lightgrey",width=.7) + geom_boxplot(width=.3)+
  theme_bw()+
  facet_grid(scenario~time) + labs(x="",y="AUC")+ theme(legend.position = "none")+
  theme(panel.grid.major = element_blank(),
        axis.title.x = element_text( size=20),
        axis.title.y = element_text( size=20),
        axis.text.x = element_text( 
          size=15),
        axis.text.y = element_text(size=20)) +
  theme(strip.text.x = element_text(size = 15)) + 
  theme(strip.text.y = element_text(size = 15)) + theme_bw(base_size = 18)+  theme(legend.position = "none")

p = ggplot(all_cases, aes(factor(method), auc,fill=method))

p +  geom_boxplot()+
  theme_bw()+
  facet_grid(scenario~time) + 
  stat_compare_means(comparisons = my_comparisons, method = "t.test") +
  labs(x="",y="AUC")+ theme(legend.position = "none")+
  theme(panel.grid.major = element_blank(),
        axis.title.x = element_text( size=15),
        axis.title.y = element_text( size=15),
        axis.text.x = element_text( 
          size=15),
        axis.text.y = element_text(size=15)) +
  theme(strip.text.x = element_text(size = 15)) + 
  theme(strip.text.y = element_text(size = 15)) + theme_bw(base_size = 18)+  theme(legend.position = "none")


my_comparisons <- list( c("cox", "po1"), c("cox", "po2"), c("cox", "po3"),c("cox", "po4") )

p +  geom_boxplot()+
  theme_bw()+
  facet_grid(scenario~time) + 
  stat_compare_means( method = "t.test",ref.group = "cox") +
  labs(x="",y="AUC")+ theme(legend.position = "none")+
  theme(panel.grid.major = element_blank(),
        axis.title.x = element_text( size=20),
        axis.title.y = element_text( size=20),
        axis.text.x = element_text( 
          size=15),
        axis.text.y = element_text(size=20)) +
  theme(strip.text.x = element_text(size = 15)) + 
  theme(strip.text.y = element_text(size = 15)) + theme_bw(base_size = 18)+  theme(legend.position = "none")
