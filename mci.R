#' Calculate marginal contribution feature importance
#'
#' This function calculates the MCI of all features within X according to random forests accuracy as an evaluation function.
#'
#' @param X A matrix or dataframe of explanatory features
#' @param y A numeric or factor vector
#' @param k Assumes soft k-size submodularity  
#' @return A numeric vector of feature importance scores
#' @export
mci<-function(X,y,k=2){
  colvec<-1:ncol(X)
  CompleteSet<-rje::powerSet(colvec,m=k+1)
  CompleteSetErrors<-rep(0,length(CompleteSet))
  
  for(e in 1:length(CompleteSetErrors)){
    if(length(CompleteSet[[e]])>0){
      rfmod<-ranger::ranger(y=y,x=as.data.frame(X[,CompleteSet[[e]]]),num.trees = 100)
      if(is.numeric(y)) CompleteSetErrors[e]<-rfmod$r.squared
      if(is.factor(y)) CompleteSetErrors[e]<- 1- rfmod$prediction.error
    }
  }
  
  if(is.numeric(y)) CompleteSetErrors[CompleteSetErrors<0]<-0
  if(is.factor(y)) CompleteSetErrors[CompleteSetErrors<0.5]<- 0.5
  
  OUTPUT<-rep(0,ncol(X))
  for(j in 1:ncol(X)){
    jsHERE<-unlist(lapply(CompleteSet, is.element,el=j))
    jSET<-CompleteSet[jsHERE]
    
    NOjSET<-lapply(jSET, setdiff,y=j)
    NOjSET<-intersect(NOjSET,CompleteSet)
    jSET<-lapply(NOjSET, c,j)
    jSET<-lapply(jSET, sort)
    
    charlistjSET<-unlist(lapply(jSET,paste,collapse=" "))
    charlistNOjSET<-unlist(lapply(NOjSET,paste,collapse=" "))
    charlistCompleteSet<-unlist(lapply(CompleteSet,paste,collapse=" "))
    errorWITH<-CompleteSetErrors[order(match(charlistCompleteSet, charlistjSET),na.last = NA)]
    errorWITHOUT<-CompleteSetErrors[order(match(charlistCompleteSet, charlistNOjSET),na.last = NA)]
    
    OUTPUT[j]<-max(errorWITH- errorWITHOUT)
  }
  OUTPUT
}



#' Calculate marginal contribution feature importance
#'
#' This function calculates the MCI of all features within X in parallel according to random forests accuracy as an evaluation function.
#'
#' @param X A matrix or dataframe of explanatory features
#' @param y A numeric or factor vector
#' @param k Assumes soft k-size submodularity 
#' @return A numeric vector of feature importance scores
#' @export
mci_par<-function(X,y,k=2){
  colvec<-1:ncol(X)
  CompleteSet<-rje::powerSet(colvec,m=k)
  
  CompleteSetErrors<-foreach::foreach(e=1:length(CompleteSet),  .inorder = FALSE,
                             .packages = c("ranger", "doParallel"),.combine = 'c')%dopar%{
                               if(length(CompleteSet[[e]])>0){
                                 rfmod<-ranger::ranger(y=y,x=as.data.frame(X[,CompleteSet[[e]]]),num.trees = 100)
                                 if(is.numeric(y)) return(rfmod$r.squared)
                                 if(is.factor(y)) return(1- rfmod$prediction.error)
                               }
                             }
  
  CompleteSetErrors<-c(0,CompleteSetErrors) #add accuracy for no features
  
  if(is.numeric(y)) CompleteSetErrors[CompleteSetErrors<0]<-0
  if(is.factor(y)) CompleteSetErrors[CompleteSetErrors<0.5]<- 0.5
  
  OUTPUT<-rep(0,ncol(X))
  for(j in 1:ncol(X)){
    jsHERE<-unlist(lapply(CompleteSet, is.element,el=j))
    jSET<-CompleteSet[jsHERE]
    
    NOjSET<-lapply(jSET, setdiff,y=j)
    NOjSET<-intersect(NOjSET,CompleteSet)
    jSET<-lapply(NOjSET, c,j)
    jSET<-lapply(jSET, sort)
    
    charlistjSET<-unlist(lapply(jSET,paste,collapse=" "))
    charlistNOjSET<-unlist(lapply(NOjSET,paste,collapse=" "))
    charlistCompleteSet<-unlist(lapply(CompleteSet,paste,collapse=" "))
    errorWITH<-CompleteSetErrors[order(match(charlistCompleteSet, charlistjSET),na.last = NA)]
    errorWITHOUT<-CompleteSetErrors[order(match(charlistCompleteSet, charlistNOjSET),na.last = NA)]
    
    OUTPUT[j]<-max(errorWITH- errorWITHOUT)
  }
  OUTPUT
}