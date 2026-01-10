{% test positive_values(model, column_name) %}

/*
    Test that all values in a column are positive (> 0).
*/

select
    {{ column_name }} as failing_value
from {{ model }}
where {{ column_name }} <= 0

{% endtest %}
