<script>
function updateStats() {
    fetch("/stats")
    .then(response => response.json())
    .then(data => {
        document.getElementById("totalSlots").innerText = data.total;
        document.getElementById("freeSlots").innerText = data.free;
        document.getElementById("occupiedSlots").innerText = data.occupied;
    });
}

setInterval(updateStats, 1000);
</script>
