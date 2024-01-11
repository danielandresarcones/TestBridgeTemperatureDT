import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
from matplotlib.widgets import Slider
from TestBridgeTemperatureDT.dummy_model import DummyModel

class BiasedDummyModel(DummyModel):
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

    def calculate_temperature(self, temp, wind_speed, sun_zenith, sun_azimuth, sun_elevation, x, y):
        return (self.temp_const * temp - self.wind_const * wind_speed - 273)*np.dot(np.sqrt((1-x)).reshape(-1,1),(1-y).reshape(1,-1)) + 273.0


    def calculate_grid(self, temp, wind_speed, sun_zenith, sun_azimuth, sun_elevation):
        """

        Calculates the temperature grid based on the given parameters.

        Parameters:
        -----------
        temp : numpy.ndarray
            The current temperature vector with shape (num_timesteps,).
        wind_speed : float
            The current wind speed.
        sun_zenith : float
            The sun zenith angle in degrees.
        sun_azimuth : float
            The sun azimuth angle in degrees.
        sun_elevation : float
            The sun elevation angle in degrees.
        """
        num_timesteps = len(temp)
        self.t_size = num_timesteps
        # Create 3D array to store temperature values for each point at each timestep
        self.grid = np.zeros((self.x_size+1, self.y_size+1, num_timesteps))

        # Calculate temperature at each point for current timestep
        for t in range(num_timesteps):
            # Interpolate temperature values to the rest of the grid
            x = np.arange(self.x_size+1)
            y = np.arange(self.y_size+1)
            self.grid[:, :, t] = self.calculate_temperature(temp[t], wind_speed[t], sun_zenith[t], sun_azimuth[t], sun_elevation[t],x/self.x_size,y/self.y_size)
    
    def update_grid(self, temp_const, wind_const, zenith_constant, azimuth_constant, elevation_constant):
        self.temp_const = temp_const
        self.wind_const = wind_const
        self.zenith_constant = zenith_constant
        self.azimuth_constant = azimuth_constant
        self.elevation_constant = elevation_constant
        self.calculate_grid(self.temp, self.wind_speed, self.sun_zenith, self.sun_azimuth, self.sun_elevation)
        return self.grid
    
    def update_parameters(self, temp_const=0.0, wind_const=0.0, zenith_constant=0.0, azimuth_constant=0.0, elevation_constant=0.0):
        self.temp_const = temp_const
        self.wind_const = wind_const
        self.zenith_constant = zenith_constant
        self.azimuth_constant = azimuth_constant
        self.elevation_constant = elevation_constant

    def store_inputs(self, temp, wind_speed, sun_zenith, sun_azimuth, sun_elevation):
        self.temp = temp
        self.wind_speed = wind_speed
        self.sun_zenith = sun_zenith
        self.sun_azimuth = sun_azimuth
        self.sun_elevation = sun_elevation

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

    def plot_with_slider(self):
        fig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.25)  # Adjust the bottom to make room for the slider

        im = ax.imshow(self.grid[:, :, 0], cmap='coolwarm', vmin=np.min(self.grid), vmax=np.max(self.grid))

        ax_slider = plt.axes([0.1, 0.01, 0.65, 0.03], facecolor='lightgoldenrodyellow')  # Define the slider's position

        slider = Slider(ax_slider, 'Time', 0, self.t_size - 1, valinit=0)
        colorbar = fig.colorbar(im)
        vmin = 273.0
        vmax = 280.0
        im.set_clim(vmin, vmax)  # Set the colorbar limits

        def update(val):
            i = int(slider.val)
            im.set_data(self.grid[:, :, i])
            ax.set_title(f'Time: {i}')
            fig.canvas.draw_idle()

        slider.on_changed(update)

        plt.show()

    @staticmethod
    def wrapper_salib(X, func=update_grid):
        """
        Wrapper function for SALib analysis. Evaluates the given function for each row in the input array X.

        Parameters:
        -----------
        X : numpy.ndarray
            Input array of shape (N, D) where N is the number of samples and D is the number of parameters.
        func : callable
            Function to be evaluated for each row in X. The function should take D arguments.

        Returns:
        --------
        numpy.ndarray
            Array of shape (N,) containing the results of evaluating func for each row in X.
        """
        N, D = X.shape
        results = np.empty(N)
        for i in range(N):
            temp_const, wind_const, zenith_constant, azimuth_constant, elevation_constant = X[i, :]
            results[i] = func(temp_const, wind_const, zenith_constant, azimuth_constant, elevation_constant)

        return results
