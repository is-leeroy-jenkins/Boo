/*
    Boo Documentation JavaScript

    Purpose:
        Reserved for small documentation-only enhancements. The file is intentionally minimal so
        the documentation site remains static, fast, and easy to maintain.
*/

document$.subscribe(function() {
    const tables = document.querySelectorAll("article table:not([class])");
    tables.forEach(function(table) {
        table.setAttribute("role", "table");
    });
});
