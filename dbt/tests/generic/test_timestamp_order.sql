{% test timestamp_order(model, start_column, end_column) %}

/*
    Test that end timestamps are after start timestamps.
*/

select
    {{ start_column }} as start_time,
    {{ end_column }} as end_time
from {{ model }}
where {{ end_column }} < {{ start_column }}

{% endtest %}
