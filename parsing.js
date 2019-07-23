const fs = require("fs");
const demofile = require("demofile");

const createCsvWriter = require('csv-writer').createObjectCsvWriter;

//For writing to file that contains deaths
const writerToDeaths = createCsvWriter({
  path: 'death.csv',
  header: [
    {id: 'round', title: 'Round'},
    {id: 'tick', title: 'Tick'},
    {id: 'victim', title: 'Victim'},
    {id: 'victimPosition', title: 'VictimPosition'},
    {id: 'attacker', title: 'Attacker'},
    {id: 'attackerPosition', title: 'AttackerPosition'},
  ]
});

//For writing to file that contains positions
const writerToPositions = createCsvWriter({
  path: 'positions.csv',
  header: [
    {id: 'round', title: 'Round'},
    {id: 'tick', title: 'Tick'},
    {id: 'player', title: 'Player'},
    {id: 'playerPosition', title: 'PlayerPosition'},
	{id: 'velocity', title: 'Velocity'},
  ]
});


fs.readFile("2093_fnatic-virtus-pro_de_cbble.dem", (err, buffer) => {
  const demoFile = new demofile.DemoFile();
  
  let deathData = [];
  let positionData = [];
  
  demoFile.gameEvents.on("player_footstep", s => {
	const player = demoFile.entities.getByUserId(s.userid);
	const playerName = player ? player.name : "unnamed";
	
	positionData.push(
		{
			tick: demoFile.currentTick,
			player: playerName,
			playerPosition: vectorToString(player.position),
			velocity: vectorToString(player.velocity)
		}
	);	
  });

  demoFile.gameEvents.on("round_officially_ended", e => {
    const teams = demoFile.teams;

    const terrorists = teams[2];
    const cts = teams[3];
	
	const finishedRound = demoFile.gameRules.roundsPlayed;  
	
	console.log(`ROUND >>${finishedRound}<< HAS ENDED!`);
	
	deathData.forEach((kill) => {
		kill.round = finishedRound;
	})
	
	positionData.forEach((pos) => {
		pos.round = finishedRound;
	})
	
   writerToPositions
	.writeRecords(positionData)
	
   writerToDeaths
	.writeRecords(deathData)
	.then(() => console.log(">> Data of last Round written to CSV"));
	
   deathData.length = 0;
   positionData.length = 0;

    console.log(
      "\tTerrorists: %s score %d\n\tCTs: %s score %d",
      terrorists.clanName,
      terrorists.score,
      cts.clanName,
      cts.score
    );
  });

  demoFile.gameEvents.on("player_death", e => {

    const victim = demoFile.entities.getByUserId(e.userid);
    const victimName = victim ? victim.name : "unnamed";

    // Attacker may have disconnected so be aware.
    // e.g. attacker could have thrown a grenade, disconnected, then that grenade
    // killed another player.
    const attacker = demoFile.entities.getByUserId(e.attacker);
    const attackerName = attacker ? attacker.name : "unnamed";

    const headshotText = e.headshot ? " HS" : "";

	deathData.push(
		{
			tick: demoFile.currentTick,
			victim: victimName,
			victimPosition: vectorToString(victim.position),
			attacker: attackerName,
			attackerPosition: vectorToString(attacker.position)

		}
	);

    console.log(`||${demoFile.currentTick}|| ${attackerName} [${e.weapon}${headshotText}] ${victimName}`);
  });

  //console.log(demoFile);

  demoFile.parse(buffer);
});

const vectorToString = (vector) => {
	const x = vector.x;
	const y = vector.y;
	const z = vector.z;

	return `"${x},${y},${z}"`
};