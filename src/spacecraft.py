class Spacecraft:
    def __init__(self, mass, r_com, I_body, I_principal, surfaces):
        self.mass = mass
        self.r_com = r_com
        self.I_body = I_body
        self.I_principal = I_principal
        self.surfaces = surfaces
