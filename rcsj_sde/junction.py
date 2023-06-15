import dataclasses
from dataclasses import dataclass
import numpy as np
import json

from rcsj_sde.utils import hbar_over_2e, kB


@dataclass
class JosephsonJunction:
    """
    Stores the circuit parameters of the Josephson junction.
    
    Current-phase relation: I_super = Ic * (a*sin(phi) + b*sin(phi/2))
    
    Attributes
    ----------
    Ic : float
        Prefactor of the current-phase relation
    a : float
        Prefactor of trivial term in the current-phase relation, a*sin(phi)
    b : float
        Prefactor of topological term in the current-phase relation, b*sin(phi/2)  
    R : float
        Parallel resistance value
    C : float
        Parallel capacitance value
    T : float
        Temperature, used for evaluating the noise term in the SDE
    t_c : float
        Characteristic time
    gamma : float
        Gamma parameter
    beta : float
        Stewart-McCumber parameter
    omega_p : float
        Plasma frequency
    epsilon : float
        Prefactor of the noise term, sigma/beta
    """

    Ic: float
    a: float
    b: float
    R: float
    C: float
    T: float

    def __post_init__(self):
        self.t_c = hbar_over_2e/(self.R*self.Ic)
        self.gamma = (self.R*self.Ic)/hbar_over_2e 
        self.beta =  self.Ic*self.R**2*self.C/hbar_over_2e
        self.omega_p = np.sqrt(self.Ic/(self.C*hbar_over_2e))
        self.epsilon = np.sqrt(2*kB*self.T/(self.Ic*hbar_over_2e))/self.beta
    
    @classmethod
    def from_json(cls, jj_str: str) -> "JosephsonJunction":
        """
        Create a JosephsonJunction from a JSON string.
        
        Parameters
        ----------
        jj_str : str
            JSON string defining the JosephsonJunction

        Returns
        -------
        JosephsonJunction
            JosephsonJunction instance
        """
        return cls(**json.loads(jj_str))     
            
    def to_json(self) -> str:
        """
        Generate JSON string from the JosephsonJunction.

        Returns
        -------
        str
            JSON string representing the JosephsonJunction
        """
        return json.dumps(dataclasses.asdict(self))
   