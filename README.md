# Ultra-marginal Feature Importance: Learning from Data with Causal Guarantees
## Contains:
1.  UMFI_aistats_an.R is R code for reproducing the results of the UMFI paper (AISTATS 2023, https://arxiv.org/abs/2204.09938). 
2.  Easily downloadable R package for basic UMFI functions (see README below for a quick tutorial)
3.  Python pacakge is currently under development

## How to use R package
``` R
library(devtools)
install_github("HydroML/UMFI",upgrade = F,dependencies = T)
library(UMFI)
data("BRCA")
X_dat<-BRCA[,2:51]
protected_col<-3

# try removing dependencies via linear regression
S<-preprocess_lr(dat=X_dat,protect = protected_col)
cor(S,S[,protected_col])

# try removing dependencies via linear regression
S<-preprocess_ot(dat=X_dat,protect = protected_col)
cor(S,S[,protected_col])

# try calculating UMFI values for the BRCA dataset (with linear regression)
BRCA$BRCA_Subtype_PAM50<-as.factor(BRCA$BRCA_Subtype_PAM50)
fi<-umfi(X_dat,BRCA$BRCA_Subtype_PAM50,mod_meth = "lr")


# try calculating UMFI values in parallel for the BRCA dataset (with optimal transport)
doParallel::registerDoParallel(cores = 12)
cl <- parallel::makeCluster(12)
doParallel::registerDoParallel(cl)

BRCA$BRCA_Subtype_PAM50<-as.factor(BRCA$BRCA_Subtype_PAM50)
fi<-umfi_par(X_dat,BRCA$BRCA_Subtype_PAM50,mod_meth = "ot")

parallel::stopCluster(cl)


# try calculating MCI values for the BRCA dataset
BRCA$BRCA_Subtype_PAM50<-as.factor(BRCA$BRCA_Subtype_PAM50)
fi<-mci(X_dat[,1:10],BRCA$BRCA_Subtype_PAM50,k=3) #warning: due to computational complexity, do not try large datasets or large k



# try calculating UMFI values in parallel for the BRCA dataset (with optimal transport)
doParallel::registerDoParallel(cores = 12)
cl <- parallel::makeCluster(12)
doParallel::registerDoParallel(cl)

BRCA$BRCA_Subtype_PAM50<-as.factor(BRCA$BRCA_Subtype_PAM50)
fi<-mci_par(X_dat,BRCA$BRCA_Subtype_PAM50,k=2) #warning: computationally expensive

parallel::stopCluster(cl)

```

