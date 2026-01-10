{% test valid_coordinates(model, lat_column, lon_column) %}

/*
    Test that coordinates are within Nashville metro bounds.
*/

select
    {{ lat_column }} as latitude,
    {{ lon_column }} as longitude
from {{ model }}
where {{ lat_column }} < {{ var('nashville_lat_min') }}
   or {{ lat_column }} > {{ var('nashville_lat_max') }}
   or {{ lon_column }} < {{ var('nashville_lon_min') }}
   or {{ lon_column }} > {{ var('nashville_lon_max') }}

{% endtest %}
