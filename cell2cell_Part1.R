###############################################################################
### Script: cell2cell_Analysis_Part1.R
### Copyright (c) 2020 by Alan Montgomery. Distributed using license CC BY-NC 4.0
### To view this license see https://creativecommons.org/licenses/by-nc/4.0/
###
### This script builds a predictive model for cell2cell that estimates both
### a logistic regression model and a classification and regression tree (CART)
### use for Part 1 of the exercise.
###
### Without making any changes to this script you can come up with a good
### tree and logistic regression model.  You can improve the decision
### tree by changing the complexity or depth of the tree, and the logistic
### regression by choosing the "right" variables -- perhaps using stepwise
### variable selection or using the variables selected in your decision tree
### However, the purpose of Part 1 of the exercise is to focus on translating
### your model, and this is where you should focus your efforts.
###
###############################################################################



###############################################################################
### setup the environment
###############################################################################

# setup environment, for plots
if (!require(reshape2)) {install.packages("reshape2"); library(reshape2)}
if (!require(gplots)) {install.packages("gplots"); library(gplots)}
if (!require(ggplot2)) {install.packages("ggplot2"); library(ggplot2)}
# data manipulation
if (!require(plyr)) {install.packages("plyr"); library(plyr)}
# setup environment, make sure this library has been installed
if (!require(tree)) {install.packages("tree"); library(tree)}
# setup environment (if you want to use fancy tree plots)
if (!require(rpart)) {install.packages("rpart"); library(rpart)}
if (!require(rattle)) {install.packages("rattle"); library(rattle)}
if (!require(rpart.plot)) {install.packages("rpart.plot"); library(rpart.plot)}
if (!require(RColorBrewer)) {install.packages("RColorBrewer"); library(RColorBrewer)}
if (!require(party)) {install.packages("party"); library(party)}
if (!require(partykit)) {install.packages("partykit"); library(partykit)}
# a better scatterplot matrix routine
if (!require(car)) {install.packages("car"); library(car)}
# better summary tables
if (!require(psych)) {install.packages("psych"); library(psych)}
# tools for logistic regression
if (!require(visreg)) {install.packages("visreg"); library(visreg)}  # visualize regression
if (!require(ROCR)) {install.packages("ROCR"); library(ROCR)}  # ROC curve for tree or logistic
if (!require(plotmo)) {install.packages("plotmo"); library(plotmo)}  # show model response

# define a function to summary a classification matrix we will use later
# this is a function which has three arguments:
#    predprob  = vector of prediction probabilities
#    predclass = vector of predicted labels
#    trueclass = vector of true class labels
confmatrix.summary <- function(predprob,predclass,trueclass) {
  # compute confusion matrix (columns have truth, rows have predictions)
  results = xtabs(~predclass+trueclass)  
  # compute usual metrics from the confusion matrix
  accuracy = (results[1,1]+results[2,2])/sum(results)   # how many correct guesses along the diagonal
  truepos = results[2,2]/(results[1,2]+results[2,2])  # how many correct "churn" guesses
  precision = results[2,2]/(results[2,1]+results[2,2]) # proportion of correct positive guesses 
  trueneg = results[1,1]/(results[2,1]+results[1,1])  # how many correct "non-churn" guesses 
  # compute the lift using the predictions for the 10% of most likely
  topchurn = as.vector( predprob >= as.numeric(quantile(predprob,probs=.9)))  # which customers are most likely to churn
  ( baseconv=sum(trueclass==1)/length(trueclass) )  # what proportion would we have expected purely due to chance
  ( actconv=sum(trueclass[topchurn])/sum(topchurn))  # what proportion did we actually predict
  ( lift=actconv/baseconv )  # what is the ratio of how many we got to what we expected
  return(list(confmatrix=results,accuracy=accuracy,truepos=truepos,precision=precision,trueneg=trueneg,lift=lift))
}



###############################################################################
### read in the data and prepare the dataset for analysis
###############################################################################

# set your working directory to where your data is stored
setwd("~/Documents/class/analytical marketing/cases/cell2cell/data")  # !! change this to your directory !!

# import dataset from file
cell2cell=read.csv("cell2cell_data.csv")
cell2celldoc=read.csv("cell2cell_doc.csv",as.is=TRUE)  # just read in as strings not factors
cell2cellinfo=cell2celldoc$description  # create a separate vector with just the variable description
rownames(cell2celldoc)=cell2celldoc$variable   # add names so we can reference like cell2celldoc["Revenue",]
names(cell2cellinfo)=cell2celldoc$variable  # add names so we can reference like cell2cellinfo["Revenue]

# set the random number seed so the samples will be the same if regenerated
set.seed(1248765792)

# prepare new values
trainsample=(cell2cell$Sample==1)  # use this sample for training
validsample=(cell2cell$Sample==2)  # use this sample for validation (e.g. comparing models)
predsample=(cell2cell$Sample==3)   # use this sample to judge accuracy of your final model
plotsample=sample(1:nrow(cell2cell),200)  # use this small sample for plotting values (otherwise plots are too dense)

# remove sample from the cell2cell set, since we have the sample variables
cell2cell$Sample=NULL

# recode the location so that we only keep the first 3 characters of the region
# and only remember the areas with more than 800 individuals, otherwise set region to OTH for other
newcsa=strtrim(cell2cell$Csa,3)  # get the MSA which is the first 3 characters
csasize=table(newcsa)  # count number of times MSA occurs
csasizeorig=rownames(csasize)  # save the city names which are in rownames of our table
csasizename=csasizeorig  # create a copy
csasizename[csasize<=800]="OTH"  # replace the city code to other for those with fewer than 800 customers
# overwrite the original Sca variable with the newly recoded variable using mapvalues
cell2cell$Csa=factor(mapvalues(newcsa,csasizeorig,csasizename))  # overwrites original Csa

# create a missing variable for age1 and age2
cell2cell$Age1[cell2cell$Age1==0]=NA  # replace zero ages with missing value
cell2cell$Age2[cell2cell$Age2==0]=NA  # replace zero ages with missing value
cell2cell$Age1miss=ifelse(is.na(cell2cell$Age1),1,0)  # create indicator for missing ages
cell2cell$Age2miss=ifelse(is.na(cell2cell$Age2),1,0)  # create indicator for missing ages

# replace missing values with means
nvarlist = sapply(cell2cell,is.numeric)  # get a list of numeric variables
NA2mean <- function(x) replace(x, is.na(x), mean(x, na.rm = TRUE))  # define a function to replace NA with means
cell2cell[nvarlist] = lapply(cell2cell[nvarlist], NA2mean)  # lapply performs NA2mean for each columns



###############################################################################
### exploratory analysis to understand the data
###############################################################################

# check number of observations in each sample
sum(trainsample)
sum(validsample)
sum(predsample)

# choose some common variables
svarlist=c("Churn","Eqpdays","Months","Recchrge","Revenue","Csa","Customer","Age1","Age2","Mailflag","Retcall")
nvarlist=c("Churn","Eqpdays","Months","Revenue","Customer","Age1")  # numeric variables only

svarlist=c("Churn","Eqpdays","Months","Recchrge","Revenue","Csa","Refurb","Uniqsubs","Mailres","Overage",
           "Mou","Setprcm","Creditde","Actvsubs","Roam","Changem","Changer","Marryno",
           "Age1","Age2","Mailflag","Retcall")

# let's take a look at just one observation
print(cell2cell[1,svarlist])

# take a look at the variable definitions
cell2cellinfo[svarlist]
#! if you every want to know what a variable is enter: cell2cellinfo["Eqpdays"], where Eqpdays is the name of your variable !

# use the describe function in the psych package to generate nicer table than "summary"
describe(cell2cell[,svarlist[-6]],fast=TRUE)   # remove the CSA variable (#6) since it is categorical
# describe the cell2cell data for Churners and Loyal customers
describeBy(cell2cell[,svarlist[-6]],group=cell2cell$Churn,fast=TRUE)

# scatterplot matrix to visualize potential relationships amongst pairs of variables
par(mfrow=c(1,1),mar=c(5,4,4,1))
pairs(cell2cell[plotsample,nvarlist])  # generates scatterplot matrix to see relationships
# nicer scatterplot matrix that color codes churners in Red, diagonals show distribution of variable given churn or not
#scatterplotMatrix(~Eqpdays+Months+Recchrge+Revenue+Customer+Age1+Age2|Churn,data=cell2cell[plotsample,])

# boxplots
par(mfrow=c(2,4),mar=c(5,5,1,1))
boxplot(Eqpdays~Churn,data=cell2cell[plotsample,],xlab="Churn",ylab="Eqpdays")
boxplot(Months~Churn,data=cell2cell[plotsample,],xlab="Churn",ylab="Months")
boxplot(Recchrge~Churn,data=cell2cell[plotsample,],xlab="Churn",ylab="Recchrge")
boxplot(Revenue~Churn,data=cell2cell[plotsample,],xlab="Churn",ylab="Revenue")
boxplot(Customer~Churn,data=cell2cell[plotsample,],xlab="Churn",ylab="Customer")
boxplot(Age1~Churn,data=cell2cell[plotsample,],xlab="Churn",ylab="Age1")
boxplot(Age2~Churn,data=cell2cell[plotsample,],xlab="Churn",ylab="Age2")
par(mfrow=c(1,1))

# cross tabs
xtabs(~Csa+Churn,data=cell2cell[trainsample,])
xtabs(~Mailflag+Churn,data=cell2cell[trainsample,])
xtabs(~Retcall+Churn,data=cell2cell[trainsample,])

# compute correlation matrix (using only complete sets of observations)
round(cor(cell2cell[,svarlist[-6]],use="pairwise.complete.obs"),digits=2)  # remove Csa (-6) since it is categorical
# here is a better visualization of the correlation matrix using a heatmap
qplot(x=Var1,y=Var2,data=melt(cor(cell2cell[,svarlist[-6]],use="p")),fill=value,geom="tile")+
  scale_fill_gradient2(limits=c(-1, 1)) + theme(axis.text.x = element_text(angle = 90, hjust = 1))



###############################################################################
### estimate a tree model with all variables
###############################################################################

# estimate a model with all the variables that is very deep
ctree.full = rpart(Churn~., data=cell2cell[trainsample,], control=rpart.control(cp=0.0005), model=TRUE)
##############
svarlist=c("Churn","Eqpdays","Months","Recchrge","Revenue","Age1","Retcall")

ctree.full = rpart(Churn~., data=cell2cell[trainsample,svarlist], control=rpart.control(cp=0.0005), model=TRUE)
#############
summary(ctree.full)
# uncomment the line below to view the full tree -- clearly needs pruning -- which is what the commands below do
#prp(ctree.full)  # make sure your plot window is large or this command can cause problems

# these lines are helpful to find the "best" value of cp
# A good choice of cp for pruning is often the leftmost value for which the mean lies below the horizontal line.
printcp(ctree.full)               # display table of optimal prunings based on complexity parameter
plotcp(ctree.full)                # visualize cross-validation results

# prune the tree back !! choose on of the lines below for treeA or treeB, and leave the other commented out !!
#ctree=prune(ctree.full,cp=0.005)  # prune tree using chosen complexity parameter !! simple tree (treeA) !!
ctree=prune(ctree.full,cp=0.00090890)  # prune tree using chosen complexity parameter !! better tree (treeB) !!

# visualize the trees 
par(mfrow=c(1,1))         # reset one graphic per panel
#plot(ctree); text(ctree)  # simple graph

prp(ctree,extra=101,nn=TRUE)  # add the size and proportion of data in the node
#fancyRpartPlot(ctree)     # fancy graphic  !! uncomment if you load library(rattle) !!
plotmo(ctree)             # evaluates selected input but holds other values at median
#plotmo(ctree,pmethod="apartdep")   # evaluates selected input and averages other values  !! pmethod="partdep" is better but slower !!

# compute predictions for the entire sample -- notice we only train on trainsample
pchurn.tree = predict(ctree,newdata=cell2cell,type='vector')
cchurn.tree = (pchurn.tree>.5)+0   # make our predictions using a 50% cutoff, !! this can threshold can be changed !!
truechurn = cell2cell$Churn

# compute confusion matrix and some usual statistics (uncomment train and prediction samples if you want to see these statistics)
#confmatrix.summary(pchurn.tree[trainsample],cchurn.tree[trainsample],truechurn[trainsample])  # summary for training sample (look to see if train is substantially better than valid)
confmatrix.summary(pchurn.tree[validsample],cchurn.tree[validsample],truechurn[validsample])  # summary for validation sample
#confmatrix.summary(pchurn.tree[predsample],cchurn.tree[predsample],truechurn[predsample]) # summary for prediction sample (use this when you want to know how good your "final" model is)

# compute ROC and AUC
rocpred.tree = prediction(pchurn.tree[validsample],cell2cell$Churn[validsample])  # compute predictions using "prediction"
rocperf.tree = performance(rocpred.tree, measure = "tpr", x.measure = "fpr")
plot(rocperf.tree, col=rainbow(10)); abline(a=0, b= 1)
auc.tmp = performance(rocpred.tree,"auc")  # compute area under curve
(auc.tree = as.numeric(auc.tmp@y.values))



###############################################################################
### estimate a logistic regression model
### you can either use a prespecified model,
### or use stepwise regression model with all the variables and their interactions
###############################################################################





### our code
# create vector of variables used in model called mvarlist, and add other variables that we want to write out
# these lines require the lrmdl and ctree to be created appropriately above
mvarlist=names(coefficients(lrmdl))[-1]   # get the variables used in your logistic regression moodel, except the intercept which is in first position
mvarlist=unlist(unique(strsplit(mvarlist,":")))   # if you have interactions then you need to uncomment this line
mvarlist.tree=names(ctree$variable.importance)   # if we want the list of variables in the tree then uncomment this line
mvarlist=unique(c(mvarlist,mvarlist.tree))  # add the variables in the tree that are not in the lr model
evarlist=c("Customer","Revenue","Churn")     # vector of extra variables to save -- regardless of whether they are in the model
varlist=c(mvarlist,evarlist)         # vector of variable names that we will use (all model variables plus ID and revenue)
print(varlist)  # vector of variables to save

# retrieve coefficients from your model
coeflist=summary(lrmdl)$coefficients  # extract coefficients estimates and std errors and z values
coefdata=data.frame(rn=rownames(coeflist),coeflist,row.names=NULL)  # change to dataframe
colnames(coefdata)=c("rn",colnames(coeflist))
print(coefdata)   # print out the coefficients
summary(coefdata)
coef_summary <- summary(lrmdl)$coefficients
print(coef_summary)

#########


# !! either use the first block of code to create your own regression,  !!
# !! or uncomment the lrmdl in the second block of code to use a prespecified model !!

# run a step-wise regression
# first estimate the null model (this just has an intercept)
##null = glm(Churn~1,data=cell2cell[trainsample,],family='binomial')
# second estimate a complete model (with all variables that you are interested in)
##full = glm(Churn~.,data=cell2cell[trainsample,],family='binomial')  # takes a long time
# if you have time uncomment the following line and include all squared terms (e.g., nonlinear effects)
##full = glm(Churn~.^2,data=cell2cell[plotsample,],family='binomial')  # takes a very long time -- but since we just want the formula for stepwise can just use plotsample instead of trainsample
# finally estimate the step wise regression starting with the null model
##lrmdl = step(null, scope=formula(full),steps=15,dir="forward")  # !! can increase beyond 15 steps, just takes more time

# this logistic regression uses some important variables and is a gives a good model
# if you uncomment the stepwise regression then comment out the following line
# (example) simpler logistic regression used in logistic regression simulator
#lrmdl=glm(Churn~Eqpdays+Retcall+Months+Overage+Mou+Changem,data=cell2cell[trainsample,],family='binomial')
# (simple) simpler logistic regression with 10 terms: stepwise model with . and steps=10
#lrmdl=glm(Churn~Eqpdays+Retcall+Months+Refurb+Uniqsubs+Mailres+Overage+Mou+Creditde+Actvsubs,data=cell2cell[trainsample,],family='binomial')
# (base: lrB) logistic regression with 20 terms and interactions: stepwise model with ^2 and steps=20, note: Eqpdays:Months means Eqpdays*Months in the model
lrmdl=glm(Churn~Eqpdays+Retcall+Months+Refurb+Uniqsubs+Mailres+Overage+Mou+Setprcm+Creditde+Actvsubs+Roam+Changem+Changer+Marryno+Age1+Eqpdays:Months+Months:Mou+Creditde:Changem+Overage:Age1,data=cell2cell[trainsample,],family='binomial')

# give a summary of the model's trained parameters
summary(lrmdl)
plotmo(lrmdl)             # evaluates selected input but holds other values at median
#plotmo(lrmdl,pmethod="partdep")   # evaluates selected input and averages other values  !! pmethod="apartdep" is faster but approximate !!

# compute predictions for the entire sample -- but model was only trained on trainsample
pchurn.lr = predict(lrmdl,newdata=cell2cell,type='response')
cchurn.lr = (pchurn.lr>.5)+0
truechurn = cell2cell$Churn

# compute confusion matrix and some usual statistics (uncomment train and prediction samples if you want to see these statistics)
#confmatrix.summary(pchurn.lr[trainsample],cchurn.lr[trainsample],truechurn[trainsample])  # summary for training sample (look to see if train is substantially better than valid)
confmatrix.summary(pchurn.lr[validsample],cchurn.lr[validsample],truechurn[validsample])  # summary for validation sample
#confmatrix.summary(pchurn.lr[predsample],cchurn.lr[predsample],truechurn[predsample]) # summary for prediction sample (use this when you want to know how good your "final" model is)

# compute ROC and AUC
rocpred.lr = prediction(pchurn.lr[validsample],cell2cell$Churn[validsample])  # compute predictions using "prediction"
rocperf.lr = performance(rocpred.lr, measure = "tpr", x.measure = "fpr")
plot(rocperf.lr, col=rainbow(10)); abline(a=0, b= 1)
auc.tmp = performance(rocpred.lr,"auc")  # compute area under curve
(auc.lr = as.numeric(auc.tmp@y.values))



###############################################################################
### compare models using ROC plot
###############################################################################

# plot all ROC curves together
plot(rocperf.lr,col="red"); abline(a=0,b=1)
plot(rocperf.tree,add=TRUE,col="blue")
legend("bottomright",c("LogRegr","Tree"),pch=15,col=c("red","blue"),bty="n")



###############################################################################
### export data for a simulator spreadsheet to "cell2cell_lrmodeldata.csv"
### uses the models that were created above, so you must have trained your models
###
### the CSV file contains the:
###  a) the model parameters from our logistic regression,
###  b) average and standard deviation of the original data,
###  c) actual data values associated with selected users
###  d) predicted probabilities of the selected users
###############################################################################

# create list of customer indices to extract for our analysis
userlist=c(15747,29301,8695,34573)

# create vector of variables used in model called mvarlist, and add other variables that we want to write out
# these lines require the lrmdl and ctree to be created appropriately above
mvarlist=names(coefficients(lrmdl))[-1]   # get the variables used in your logistic regression moodel, except the intercept which is in first position
mvarlist=unlist(unique(strsplit(mvarlist,":")))   # if you have interactions then you need to uncomment this line
mvarlist.tree=names(ctree$variable.importance)   # if we want the list of variables in the tree then uncomment this line
mvarlist=unique(c(mvarlist,mvarlist.tree))  # add the variables in the tree that are not in the lr model
evarlist=c("Customer","Revenue","Churn")     # vector of extra variables to save -- regardless of whether they are in the model
varlist=c(mvarlist,evarlist)         # vector of variable names that we will use (all model variables plus ID and revenue)
print(varlist)  # vector of variables to save

# retrieve coefficients from your model
coeflist=summary(lrmdl)$coefficients  # extract coefficients estimates and std errors and z values
coefdata=data.frame(rn=rownames(coeflist),coeflist,row.names=NULL)  # change to dataframe
colnames(coefdata)=c("rn",colnames(coeflist))
print(coefdata)   # print out the coefficients

# retrieve data about the users (assumes that pchurn.lr and pchurn.tree have been computed in earlier part of script)
userpred=cbind(pchurn.lr[userlist],pchurn.tree[userlist])  # create matrix of predictions from our model for selected users
colnames(userpred)=c("pchurn.lr","pchurn.tree")  # label our columns appropriately
modeldata=model.matrix(lrmdl,data=cell2cell[userlist,])  # construct the data used in the model
userdata=cell2cell[userlist,evarlist]  # get additional variables
userdata=t(cbind(modeldata,userdata,userpred))  # get relevant data for a set of customers
userdata=data.frame(rn=rownames(userdata),userdata,row.names=NULL)  # change to dataframe
print(userdata)   # print out user data

# retrieve averages and std dev across all users
modelall=model.matrix(lrmdl,data=cell2cell[trainsample,])  # get a matrix of all data used in the model (just training sample)
meandata=apply(modelall,2,mean) # compute the average for the selected variables (the "2" means compute by column)
sddata=apply(modelall,2,sd)  # compute the standard deviation for selected variables
descdata=data.frame(rn=names(meandata),meandata,sddata,row.names=NULL)  # merge the vectors with the mean and stddev into a single dataframe
print(descdata)   # print out the descriptive values

# combine the data together to make it easier to dump out to a single spreadsheet
mdata=join(coefdata,descdata,type='full',by='rn')  # merge the coefficients and descriptive stats
mdata=join(mdata,userdata,type='full',by='rn')  # create a final single dataframe
print(mdata)    # print out the combined data

# write the data to a spreadsheet
write.csv(mdata,file="cell2cell_lrmodeldata.csv")   # if you want you can import this file into excel for easier processing


