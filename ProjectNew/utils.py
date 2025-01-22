import numpy as np


def plane_sailing_position(start_point, bearing, distance):
    # Convert lat/lon and course to radians
    lat1, lon1 = np.radians(start_point)
    bearing = np.radians(bearing)

    # Convert distance in nautical miles to degrees of latitude/longitude
    distance_rad = np.radians(distance / 60)  # Distance in radians (1 degree = 60 NM)

    # Calculate the change in latitude (delta_lat)
    delta_lat = distance_rad * np.cos(bearing) * np.cos(lat1)

    # Calculate the new latitude
    new_lat = lat1 + delta_lat

    # Calculate the mean latitude (average of the original and new latitude)
    mean_lat = (lat1 + new_lat) / 2

    # Calculate the change in longitude (delta_lon)
    if np.cos(mean_lat) != 0:
        delta_lon = distance_rad * np.sin(bearing) / np.cos(mean_lat)
    else:
        delta_lon = 0

    # Calculate the new longitude
    new_lon = lon1 + delta_lon

    # Convert the new latitude and longitude back to degrees
    new_lat = np.degrees(new_lat)
    new_lon = np.degrees(new_lon)

    return np.array([new_lat, new_lon])


def plane_sailing_next_position(start_point, course, speed, time_interval=6):
    """
    Calculates the next position based on a starting point, course, and distance using plane sailing approximation.

    :param time_interval: time interval in minutes
    :param start_point: tuple (lat, lon) in degrees
    :param course: course (bearing) in degrees
    :param distance: distance in nautical miles
    :return: tuple (new_lat, new_lon) in degrees
    """
    # Convert lat/lon and course to radians
    lat1, lon1 = np.radians(start_point)
    course = np.radians(course)

    # Calculate distance
    distance = speed * time_interval / 60

    # Convert distance in nautical miles to degrees of latitude/longitude
    distance_rad = np.radians(distance / 60)  # Distance in radians (1 degree = 60 NM)

    # Calculate the change in latitude (delta_lat)
    delta_lat = distance_rad * np.cos(course)

    # Calculate the new latitude
    new_lat = lat1 + delta_lat

    # Calculate the mean latitude (average of the original and new latitude)
    mean_lat = (lat1 + new_lat) / 2

    # Calculate the change in longitude (delta_lon)
    if np.cos(mean_lat) != 0:
        delta_lon = distance_rad * np.sin(course) / np.cos(mean_lat)
    else:
        delta_lon = 0

    # Calculate the new longitude
    new_lon = lon1 + delta_lon

    # Convert the new latitude and longitude back to degrees
    new_lat = np.degrees(new_lat)
    new_lon = np.degrees(new_lon)

    return np.array([new_lat, new_lon])


def mercator_latitude(lat):
    return np.log(np.tan(np.pi / 4 + lat / 2))


def mercator_conversion(lat1, lon1, lat2, lon2):
    # Convert degrees to radians
    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)
    lon1 = np.radians(lon1)
    lon2 = np.radians(lon2)

    delta_phi = mercator_latitude(lat2) - mercator_latitude(lat1)

    # Difference in longitudes
    delta_lambda = lon2 - lon1

    return delta_phi, delta_lambda


def rumbline_distance(start_point, end_point):

    lat1, lon1 = start_point
    lat2, lon2 = end_point

    # Use the midpoint latitude to minimize cosine variation
    mid_lat = (lat1 + lat2) / 2
    delta_phi, delta_lambda = mercator_conversion(lat1, lon1, lat2, lon2)

    return np.sqrt((delta_lambda * np.cos(np.radians(mid_lat))) ** 2 + delta_phi ** 2) * 3440.065


def calculate_bearing(lat1, lon1, lat2, lon2):
    """
    Calculate the bearing from the current position to the waypoint.
    """
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    delta_lon = lon2 - lon1
    x = np.sin(delta_lon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(delta_lon)
    bearing = np.degrees(np.arctan2(x, y))
    return (bearing + 360) % 360  # Normalize to [0, 360]


def calculate_relative_bearing(heading, bearing):
    """
    Calculates the relative bearing between the ship's heading and a given true bearing.
    The result is expressed in a semi-circle coordinate system:
    - Starboard bearings (clockwise from the heading) are positive (0 to 180 degrees).
    - Port bearings (counterclockwise from the heading) are negative (0 to -180 degrees).

    :param heading: Current ship heading in degrees (0 to 360).
    :param bearing: True bearing to the target in degrees (0 to 360).
    :return: Relative bearing in degrees (positive for starboard, negative for port).
    """
    # Normalize heading and bearing to 0-360 degrees
    heading = heading % 360
    bearing = bearing % 360

    # Calculate the difference between the bearing and the heading
    diff = (bearing - heading + 360) % 360

    # Map the difference to the semi-circle system
    if diff > 180:
        relative_bearing = diff - 360  # Convert to negative (port side)
    else:
        relative_bearing = diff  # Positive (starboard side)

    return relative_bearing


if __name__ == '__main__':
    true_bearing = calculate_bearing(30.1, 100.1, 30.4, 100.25)
    print(true_bearing)
    print(calculate_relative_bearing(45, true_bearing))
