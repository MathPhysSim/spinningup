import math

class twissElement():

    def __init__(self,beta,alpha,d,mu):
        self.beta = beta
        self.alpha = alpha
        self.mu = mu

def transport(element1,element2,x,px):
    mu = element2.mu - element1.mu
    alpha1 = element1.alpha
    alpha2 = element2.alpha
    beta1 = element1.beta
    beta2 = element2.beta


    m11 = math.sqrt(beta2/beta1)*(math.cos(mu)+alpha1*math.sin(mu))
    m12 = math.sqrt(beta1*beta2)*math.sin(mu)
    m21 = ((alpha1-alpha2)*math.cos(mu)-(1+alpha1*alpha2)*math.sin(mu))/math.sqrt(beta1*beta2)
    m22 = math.sqrt(beta1/beta2)*(math.cos(mu)-alpha2*math.sin(mu))

    return m11*x+m12*px, m21*x+m22*px