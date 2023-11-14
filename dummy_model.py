
import numpy as np
import xarray as xr

class DummyModel:
    """
    A dummy model for estimating bridge temperature.

    Attributes:
    -----------
    temp_const : float
        A constant for temperature calculation.
    wind_const : float
        A constant for wind speed calculation.
    x_size : int
        The size of the x-axis grid.
    y_size : int
        The size of the y-axis grid.
    grid : numpy.ndarray
        A 2D array representing the grid of temperature values.

    Methods:
    --------
    calculate_temperature(temp, wind_speed, sun_zenith, sun_azimuth, sun_elevation)
        Calculates the temperature based on the given parameters.
    get_temperature(x, y)
        Returns the temperature value at the given coordinates.
    calculate_grid(temp, wind_speed)
        Calculates the temperature grid based on the given parameters.
    output_to_xdmf(filename)
        Outputs the temperature grid to an XDMF file.
    """

    def __init__(self, temp_const, wind_const, x_size, y_size):
        self.temp_const = temp_const
        self.wind_const = wind_const
        self.x_size = x_size
        self.y_size = y_size
        self.grid = np.zeros((x_size, y_size))

    def calculate_temperature(self, temp, wind_speed, sun_zenith, sun_azimuth, sun_elevation):
        return self.temp_const * temp + self.wind_const * wind_speed

    def get_temperature(self, x, y):
        """
        Returns the temperature value at the given coordinates.

        Parameters:
        -----------
        x : int
            The x-coordinate.
        y : int
            The y-coordinate.

        Returns:
        --------
        float
            The temperature value at the given coordinates.
        """
        return self.grid[x, y]

    def calculate_grid(self, temp, wind_speed, sun_zenith, sun_azimuth, sun_elevation):
        """
        Calculates the temperature grid based on the given parameters.

        Parameters:
        -----------
        temp : float
            The current temperature.
        wind_speed : float
            The current wind speed.
        """
        for i in range(self.x_size):
            for j in range(self.y_size):
                self.grid[i, j] = self.calculate_temperature(temp, wind_speed, sun_zenith, sun_azimuth, sun_elevation)

    def output_to_xdmf(self, filename):
        """
        Outputs the temperature grid to an XDMF file.

        Parameters:
        -----------
        filename : str
            The name of the output file.
        """
        x = np.arange(self.x_size)
        y = np.arange(self.y_size)
        coords = {'x': x, 'y': y}
        data_vars = {'temperature': (['x', 'y'], self.grid)}
        ds = xr.Dataset(data_vars, coords)
        ds.to_netcdf(filename + '.nc')
        with open(filename + '.xdmf', 'w') as f:
            f.write('<?xml version="1.0" ?>\n')
            f.write('<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>\n')
            f.write('<Xdmf Version="2.0">\n')
            f.write('  <Domain>\n')
            f.write('    <Grid Name="grid" GridType="Uniform">\n')
            f.write('      <Topology TopologyType="2DRectMesh" NumberOfElements="%d %d"/>\n' % (self.x_size, self.y_size))
            f.write('      <Geometry GeometryType="ORIGIN_DXDY">\n')
            f.write('        <DataItem Name="Origin" Dimensions="2" NumberType="Float" Precision="4" Format="XML">0.0 0.0</DataItem>\n')
            f.write('        <DataItem Name="Spacing" Dimensions="2" NumberType="Float" Precision="4" Format="XML">1.0 1.0</DataItem>\n')
            f.write('      </Geometry>\n')
            f.write('      <Attribute Name="temperature" AttributeType="Scalar" Center="Node">\n')
            f.write('        <DataItem Dimensions="%d %d" NumberType="Float" Precision="4" Format="HDF">%s:/%s</DataItem>\n' % (self.x_size, self.y_size, filename + '.nc', 'temperature'))
            f.write('      </Attribute>\n')
            f.write('    </Grid>\n')
            f.write('  </Domain>\n')
            f.write('</Xdmf>\n')
