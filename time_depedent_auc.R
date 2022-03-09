

# time dependent AUC
# Average
#POCNN_one_time = c(0.804,0.657,0.627, 0.6)
POCNN_so = c(0.750, 0.772,0.730,0.822)
POCNN_mo = c(0.781,0.532,0.671,0.776 ) #'2_year': array([0.78152628]), '3.5_year': array([0.53232503]), '5_year': array([0.67091199]), '8_year': array([0.77603197]), 
IPCWPOCNN_so = c(0.736,0.731,0.720,0.832)
IPCWPOCNN_mo =  c(0.787,0.672,0.673,0.710)
CoxCNN = c(0.676,0.671,0.691,0.789)

auc_plot = data.frame(POCNN_so,POCNN_mo,IPCWPOCNN_so,IPCWPOCNN_mo,CoxCNN,Years=c(2,3.5,5,8))
labels = c("POCNN single output","POCNN multi-output","IPCW-POCNN single output","IPCW-POCNN multi-output", "CoxCNN","Years")
colnames(auc_plot)<-labels
auc_plot <- reshape2::melt(auc_plot, id.vars = c('Years'))
names(auc_plot) <- c('Years', 'Models', 'AUC')
# rename


ggplot(auc_plot, aes(x=Years, y=AUC, colour=Models)) + 
  geom_line(aes(linetype=Models), size=1) +
  geom_point(aes(shape=Models))   +
  labs(y='Time dependent AUC') + 
  scale_x_continuous(breaks=c(2,3.5,5,8,5)) + scale_y_continuous(breaks=seq(.5,1,.05)) + theme(panel.grid.major = element_blank() )+
  theme(panel.grid.major = element_blank(),
        axis.title.x = element_text( size=15),
        axis.title.y = element_text( size=15),
        axis.text.x = element_text( 
          size=15),
        axis.text.y = element_text(size=15)) +
  theme(strip.text.x = element_text(size = 15)) + 
  theme(strip.text.y = element_text(size = 15)) + theme_bw(base_size = 15) +  theme(legend.position = "bottom") 
                     
  
ggplot(auc_plot, aes(x=Years, y=AUC, colour=Models)) + 
  geom_line() +
  geom_point()   +
  labs(y='Time dependent AUC', colour = "")+
  scale_x_continuous(breaks=c(2,3.5,5,8,5)) + scale_y_continuous(breaks=seq(.5,1,.05)) + theme(panel.grid.major = element_blank() )+
  theme(panel.grid.major = element_blank(),
        axis.title.x = element_text( size=15),
        axis.title.y = element_text( size=15),
        axis.text.x = element_text( 
          size=15),
        axis.text.y = element_text(size=15)) +
  theme(strip.text.x = element_text(size = 15)) + 
  theme(strip.text.y = element_text(size = 15)) + theme_bw(base_size = 15) +  theme(legend.position = "bottom") + theme(legend.text = element_text(size = 7))+
  guides(colour=guide_legend(nrow=2,byrow=TRUE))

  
  
  scale_linetype_manual(values = c(1,2,1,1,2,1)) +
  scale_shape_manual(values=c(0,1,2,3,4,5))


## 75th percentile
  POCNN_so = c(0.729 , 0.747 , 0.719 , 0.814)
  POCNN_mo = c(0.777 , 0.540 , 0.664, 0.779) #{'2_year': array([0.77699621]), '3.5_year': array([0.54012858]), '5_year': array([0.66422637]), '8_year': array([0.77951126]), '10_year': array([0.73295052])})
  
  IPCWPOCNN_so = c(0.732 , 0.726 , 0.736 , 0.837)
  IPCWPOCNN_mo =  c(0.783 , 0.669 , 0.667 ,  0.727)
  CoxCNN = c(0.715 , 0.694 , 0.711 , 0.798)
  
  auc_plot = data.frame(POCNN_so,POCNN_mo,IPCWPOCNN_so,IPCWPOCNN_mo,CoxCNN,Years=c(2,3.5,5,8))
  labels = c("POCNN single output","POCNN multi-output","IPCW-POCNN single output","IPCW-POCNN multi-output", "CoxCNN","Years")
  colnames(auc_plot)<-labels
  auc_plot <- reshape2::melt(auc_plot, id.vars = c('Years'))
  names(auc_plot) <- c('Years', 'Models', 'AUC')

  ggplot(auc_plot, aes(x=Years, y=AUC, colour=Models)) + 
    geom_line() +
    geom_point()   +
    labs(y='Time dependent AUC', colour = "")+
    scale_x_continuous(breaks=c(2,3.5,5,8,5)) + scale_y_continuous(breaks=seq(.5,1,.05)) + theme(panel.grid.major = element_blank() )+
    theme(panel.grid.major = element_blank(),
          axis.title.x = element_text( size=15),
          axis.title.y = element_text( size=15),
          axis.text.x = element_text( 
            size=15),
          axis.text.y = element_text(size=15)) +
    theme(strip.text.x = element_text(size = 15)) + 
    theme(strip.text.y = element_text(size = 15)) + theme_bw(base_size = 15) +  theme(legend.position = "bottom") + theme(legend.text = element_text(size = 7))+
    guides(colour=guide_legend(nrow=2,byrow=TRUE))


  ####################################################
  
  # Average
  #POCNN_one_time = c(0.804,0.657,0.627, 0.6)
  POCNN_so = c(0.750, 0.772,0.730,0.822)
  POCNN_mo = c(0.781,0.532,0.671,0.776 ) #'2_year': array([0.78152628]), '3.5_year': array([0.53232503]), '5_year': array([0.67091199]), '8_year': array([0.77603197]), 
  IPCWPOCNN_so = c(0.736,0.731,0.720,0.832)
  IPCWPOCNN_mo =  c(0.787,0.672,0.673,0.710)
  CoxCNN = c(0.676,0.671,0.691,0.789)
  
  auc_plot = data.frame(POCNN_so,POCNN_mo,IPCWPOCNN_so,IPCWPOCNN_mo,CoxCNN,Years=c(2,3.5,5,8),criteria='Average')
  labels = c("POCNN single output","POCNN multi-output","IPCW-POCNN single output","IPCW-POCNN multi-output", "CoxCNN","Years",'criteria')
  colnames(auc_plot)<-labels
  auc_plot <- reshape2::melt(auc_plot, id.vars = c('Years','criteria'))
  names(auc_plot) <- c('Years','criteria', 'Models', 'AUC')
  
  #75th percentile
  POCNN_so = c(0.729 , 0.747 , 0.719 , 0.814)
  POCNN_mo = c(0.777 , 0.540 , 0.664, 0.779) #{'2_year': array([0.77699621]), '3.5_year': array([0.54012858]), '5_year': array([0.66422637]), '8_year': array([0.77951126]), '10_year': array([0.73295052])})
  IPCWPOCNN_so = c(0.732 , 0.726 , 0.736 , 0.837)
  IPCWPOCNN_mo =  c(0.783 , 0.669 , 0.667 ,  0.727)
  CoxCNN = c(0.715 , 0.694 , 0.711 , 0.798)
  auc_plot_p = data.frame(POCNN_so,POCNN_mo,IPCWPOCNN_so,IPCWPOCNN_mo,CoxCNN,Years=c(2,3.5,5,8),criteria='75th percentile')
  labels = c("POCNN single output","POCNN multi-output","IPCW-POCNN single output","IPCW-POCNN multi-output", "CoxCNN","Years",'criteria')
  colnames(auc_plot_p)<-labels
  auc_plot_p <- reshape2::melt(auc_plot_p, id.vars = c('Years','criteria'))
  names(auc_plot_p) <- c('Years','criteria', 'Models', 'AUC')

  all_together = rbind(auc_plot,auc_plot_p)  
  
  ggplot(all_together, aes(x=Years, y=AUC, colour=Models,fill=criteria)) + 
    geom_line(aes(linetype=criteria)) +
    geom_point()   +
    labs(y='Time dependent AUC', colour = "") +
    scale_x_continuous(breaks=c(2,3.5,5,8,5)) + scale_y_continuous(breaks=seq(.5,1,.05)) + theme(panel.grid.major = element_blank() )+
    theme(panel.grid.major = element_blank(),
          axis.title.x = element_text( size=15),
          axis.title.y = element_text( size=15),
          axis.text.x = element_text( 
            size=15),
          axis.text.y = element_text(size=15)) +
    theme(strip.text.x = element_text(size = 15)) + 
    theme(strip.text.y = element_text(size = 15)) + theme_bw(base_size = 15) +  theme(legend.position = "bottom") + theme(legend.text = element_text(size = 5))+
    guides(colour=guide_legend(nrow=2,byrow=TRUE))