import dolfinx as df
import numpy as np
import pint
import ufl
from petsc4py.PETSc import ScalarType

from fenicsxconcrete.finite_element_problem.base_material import MaterialProblem, SolutionFields, QuadratureFields
from fenicsxconcrete.experimental_setup import Experiment
from fenicsxconcrete.util import ureg
from fenicsxconcrete.experimental_setup import CantileverBeam

class ThermalProblem(MaterialProblem):

    def __init__(
        self,
        experiment: Experiment,
        parameters: dict[str, pint.Quantity],
        pv_name: str = "pv_output_full",
        pv_path: str = None,
    ) -> None:
        
        super().__init__(experiment, parameters, pv_name, pv_path)

    def setup(self) -> None:

        self.time = 0.0 * ureg("s")
        normals = df.fem.FacetNormal(self.mesh)

        # define function space ets.
        self.V = df.fem.VectorFunctionSpace(self.mesh, ("Lagrange", self.p["degree"], (self.mesh.geometry.dim,)))  # 2 for quadratic elements
        self.V_scalar = df.fem.VectorFunctionSpace(self.mesh, ("Lagrange", self.p["degree"], (1,)))

        # Define variational problem
        self.T_trial = ufl.TrialFunction(self.V)
        self.v = ufl.TestFunction(self.V)

        self.fields = SolutionFields(temperature=df.fem.Function(self.V, name="temperature"))

        # initialize L field, not sure if this is the best way...
        zero_field = df.fem.Constant(self.mesh, ScalarType(np.zeros(self.p["dim"])))
        self.L = ufl.dot(zero_field, self.v) * ufl.dx

        # apply heat sources

        # Define angle of incidence for radiation
        #TODO: This is not done yet

        # Define conduction term
        q_conduction = - self.p["k"] * ufl.dot(ufl.grad(self.fields.temperature), ufl.grad(self.v)) * ufl.dx

        # Define convection term
        # TODO: Check coordinates coherence for wind
        # TODO: Normal n must be obtained from the mesh
        h = self.calculate_h(np.sqrt(np.array(self.p["wind_speed"]).dot(np.array(self.p["wind_speed"]))), self.p["D"])
        q_convection = - h * (self.fields.temperature - self.p["T_inf"]) * ufl.dot(normals, self.v) * ufl.ds

        # Define radiation term
        q_radiation = - self.p["epsilon"] * self.p["sigma"] * (self.fields.temperature**4 - self.p["T_inf"]**4) * ufl.dot(normals, self.v) * ufl.ds

        # Define heat source term
        q_heat_source = self.p["q"] * ufl.dot(normals, self.v) * ufl.ds

        # Define total heat flux
        self.L = self.L + q_conduction + q_convection + q_radiation + q_heat_source

        # boundary conditions only after function space
        # TODO: Create temperature BCs
        # bcs = self.experiment.create_adiabatic_boundary(self.V)
        bc = df.fem.dirichletbc(self.V, ufl.Constant(300), 'near(x[1], 0)')

        self.a = ufl.inner(ufl.grad(self.T_trial), self.v) * ufl.dx

        self.weak_form_problem = df.fem.petsc.LinearProblem(
            self.a,
            self.L,
            bcs=bc,
            u=self.fields.displacement,
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        )

    @staticmethod
    def parameter_description() -> dict[str, str]:
        """static method returning a description dictionary for required parameters

        Returns:
            description dictionary

        """
        description = {
            "g": "gravity",
            "dt": "time step",
            "rho": "density of fresh concrete",
            "E": "Young's Modulus",
            "nu": "Poissons Ratio",
            "stress_state": "for 2D plain stress or plane strain",
            "degree": "Polynomial degree for the FEM model",
            "k": "thermal conductivity",
            "T_inf": "ambient temperature",
            "epsilon": "emissivity",
            "sigma": "Stefan-Boltzmann constant",
            "q": "heat source",
            "n": "normal vector",
            "wind_speed": "wind speed",
            "D": "characteristic length",
            "rho_air": "density of air",
            "mu_air": "dynamic viscosity of air"
        }

        return description

    @staticmethod
    def default_parameters() -> tuple[Experiment, dict[str, pint.Quantity]]:
        """returns a dictionary with required parameters and a set of working values as example"""
        # default setup for this material
        experiment = CantileverBeam(CantileverBeam.default_parameters())

        model_parameters = {}
        model_parameters["g"] = 9.81 * ureg("m/s^2")
        model_parameters["dt"] = 1.0 * ureg("s")

        model_parameters["rho"] = 7750 * ureg("kg/m^3")
        model_parameters["E"] = 210e9 * ureg("N/m^2")
        model_parameters["nu"] = 0.28 * ureg("")

        model_parameters["stress_state"] = "plane_strain" * ureg("")
        model_parameters["degree"] = 2 * ureg("")  # polynomial degree
        model_parameters["dt"] = 1.0 * ureg("s")

        model_parameters["k"] = 1.0 * ureg("W/(m*K)")
        model_parameters["T_inf"] = 300 * ureg("K")
        model_parameters["epsilon"] = 0.8 * ureg("")
        model_parameters["sigma"] = 5.67e-8 * ureg("W/(m^2*K^4)")
        model_parameters["q"] = 0.0 * ureg("W/m^2")
        model_parameters["wind_speed"] = np.array([0.0, 0.0, 0.0]) * ureg("m/s")
        model_parameters["D"] = 1.0 * ureg("m")
        model_parameters["rho_air"] = 1.2 * ureg("kg/m^3")
        model_parameters["mu_air"] = 1.8e-5 * ureg("Pa*s")

        return experiment, model_parameters
    
    
    def calculate_h(self, V, D):
        rho = self.p["rho_air"]  # Density of air (for example)
        mu = self.p["mu_air"] # Dynamic viscosity of air (for example)
        
        Re = V * D / mu
        Pr = 0.7  # Prandtl number for air
        
        if Re < 5e5:
            Nu = 0.664 * Re**0.5 * Pr**(1/3)
        else:
            Nu = 0.037 * Re**0.8 * Pr**(1/3)
        
        h = Nu * self.p["k"]/ D
        return h

    # Stress computation for linear elastic problem
    
    def solve(self) -> None:
        self.logger.info("solving t=%s", self.time)
        self.weak_form_problem.solve()

        # TODO Defined as abstractmethod. Should it depend on sensor instead of material?
        self.compute_residuals()

        # get sensor data
        for sensor_name in self.sensors:
            # go through all sensors and measure
            self.sensors[sensor_name].measure(self)

    def compute_residuals(self) -> None:
        self.residual = ufl.action(self.a, self.fields.temperature) - self.L

    # paraview output
    # TODO move this to sensor definition!?!?!
    def pv_plot(self) -> None:
        # Displacement Plot

        with df.io.XDMFFile(self.mesh.comm, self.pv_output_file, "a") as f:
            f.write_function(self.fields.displacement, self.time)
