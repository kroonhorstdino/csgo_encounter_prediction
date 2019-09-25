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

<<<<<<< HEAD
<<<<<<< HEAD
let writerObject = {
    path: 'parsed_files\\positions.csv'
};

const playerFeatures = [
    /*{
            id: 'playerName',
            title: 'PlayerName'
        },*/
    {
        id: 'isAlive',
        title: 'IsAlive'
    }, {
        id: 'positionX',
        title: 'PositionX'
    }, {
        id: 'positionY',
        title: 'PositionY'
    }, {
        id: 'positionZ',
        title: 'PositionZ'
    }, {
        id: 'velocityX',
        title: 'VelocityX'
    }, {
        id: 'velocityY',
        title: 'VelocityY'
    }, {
        id: 'velocityZ',
        title: 'VelocityZ'
    },
    //Relative distance to all other players (1-9)
    {
        id: 'distanceToAlly_1',
        title: 'DistanceTo1'
    }, {
        id: 'distanceToAlly_2',
        title: 'DistanceTo2'
    }, {
        id: 'distanceToAlly_3',
        title: 'DistanceTo3'
    }, {
        id: 'distanceToAlly_4',
        title: 'DistanceTo9'
    }, {
        id: 'distanceToEnemy_0',
        title: 'DistanceTo_4'
    }, {
        id: 'distanceToEnemy_1',
        title: 'DistanceTo_5'
    }, {
        id: 'distanceToEnemy_2',
        title: 'DistanceTo_6'
    }, {
        id: 'distanceToEnemy_3',
        title: 'DistanceTo_7'
    }, {
        id: 'distanceToEnemy_4',
        title: 'DistanceTo_8'
    },
]

let allFeatures = [{
        id: 'round',
        title: 'Round'
    },
    {
        id: 'tick',
        title: 'Tick'
    }
]

//Create features for each player
for (const feature of playerFeatures) {
    for (let i = 0; i < 10; i++) {
        let newId = `${i}_${feature.id}`;
        let newTitle = `${i}_${feature.id}`;

        allFeatures.push({
            id: newId,
            title: newTitle
        });
    }
}

writerObject["header"] = allFeatures;

/**
 * Colums in the CSV file that contains player positions
 */
const writerToPlayerInfo = createCsvWriter(writerObject);

class DemoFileParser {
    constructor() {
        this.demoFile = new demofile.DemoFile();
        this.playersInTeams = [];

        this.deathDataBuffer = [];
        this.playerInfoBuffer = [];
        this.currentDemoFileName = "";
        this.currentDemoFilePath = "";
    }

    parseDemoFile(pathToFile) {
        this.currentDemoFilePath = pathToFile;
        this.currentDemoFileName = path.basename(pathToFile);
        let buffer = null;

        try {
            buffer = fs.readFileSync(this.currentDemoFilePath);
        } catch (e) {
            console.error("Demo file could not be opened!" + e);
            return;
        }

        this.subscribeToDemoEvents();
        this.demoFile.parse(buffer);
    }

    /**
     * Returns which ticks are supposed to be used for parsing.
     *
     * @returns
     * @memberof DemoFileParser
     */
    getTickSampleRateModulo() {
        switch (this.demoFile.tickRate) {
            case 128:
                return 16;
                break;
            case 64:
                return 8; //Get every 8th tick
                break;
            case 32:
                return 4; //Get every 4th tick
                break;
            default:
                //In case of irregular tick rates
                return Math.round(this.demoFile.tickRate / (this.demoFile.tickRate / 8)) //64 / (64 / 8)
                break;
        }
    }

    /**
     * 
     *
     * @memberof DemoFileParser
     */
    subscribeToDemoEvents() {

        //Beginning of demo file
        this.demoFile.on("start", s => {
            this.SampleRateModulo = this.getTickSampleRateModulo();
            this.ignoreTicks = true; //Ingore ticks until first round starts

            //Put players into an objects that holds both teams
            for (const team of this.demoFile.teams) {
                this.playersInTeams.push({
                    index: team.index,
                    players: team.members,
                });
            }
        });

        //End of the demo file
        this.demoFile.on("end", e => {
            this.on_round_officially_ended();
            console.log(`Parsing of demo ${this.currentDemoFileName} is complete`);
        });

        //Get player info at each relevant tick
        this.demoFile.on("tickend", tick => this.on_tickend());

        this.demoFile.gameEvents.on("round_start", s => this.on_round_start());
        this.demoFile.gameEvents.on("round_freeze_end", f => this.on_round_freeze_end());

        this.demoFile.gameEvents.on("round_officially_ended", s => this.on_round_officially_ended());

        console.log("Succesfully subscribed to all events!")
    }

    on_tickend() {
        /** 
         * If this tick is not relevant, do not parse this tick
         */
        if (!this.isRelevantTick()) return;

        if (this.demoFile.currentTick % 2048 == 0) {
            console.log("Current Tick " + this.demoFile.currentTick);
        }

        /** 
         * Get player positions each tick
         */
        this.getAllPlayerInfo();
    }

    /**
     * Is this tick relevant for parsing?
     *
     * @returns
     * @memberof DemoFileParser
     */
    isRelevantTick() {
        return !this.ignoreTicks && (this.demoFile.currentTick % this.SampleRateModulo == 0);
    }

    /**
     * Parses information of all players for current tick
     *
     * @memberof DemoFileParser
     */
    getAllPlayerInfo() {

        //Terrorist
        let t = this.demoFile.teams[2];
        //Counter-Terrorist
        let ct = this.demoFile.teams[3];

        if (t == undefined || ct == undefined) return;

        let allPlayers = [].concat(t.members, ct.members);

        const currentTick = this.demoFile.currentTick;

        if (allPlayers.length != 10 || allPlayers.includes(null)) {
            console.log("Current Tick " + currentTick + " | Teams are not complete, discarding this tick!")
            return
        }

        //Player counter for later loop
        let playerCounter = 0;

        //Info from current tick for all players
        let info = {
            tick: currentTick
        };

        //Go through all players
        for (const player of allPlayers) {

            if (player == null) {
                playerCounter++;
                continue;
            }

            const playerID = player ? player.userId : "-1";
            const playerName = player ? player.name : "playerMissing";


            let position = player.position;

            info[playerCounter + "_playerID"] = playerID;
            //info[playerCounter + "_playerName"] = playerName;
            info[playerCounter + "_isAlive"] = player.isAlive ? 1 : 0;
            info[playerCounter + "_positionX"] = position.x;
            info[playerCounter + "_positionY"] = position.y;
            info[playerCounter + "_positionZ"] = position.z;
            info[playerCounter + "_velocityX"] = player.velocity.x;
            info[playerCounter + "_velocityY"] = player.velocity.y;
            info[playerCounter + "_velocityZ"] = player.velocity.z;

            //Distance to other allies (index beings at 1, 0 is the current player)
            let i = 0;

            for (const otherPlayer of t.members) {

                if (otherPlayer == null) {
                    i++;
                    continue;
                }

                //If this is the same player
                if (otherPlayer.userId == player.userId) {
                    i++;
                    continue;
                };

                let distance = this.calcVectorDistance(player.position, otherPlayer.position);

                //If in enemy team
                if (player.teamNumber != otherPlayer.teamNumber) {
                    info[playerCounter + "_distanceToEnemy_".concat(i)] = distance;
                }
                //If on allied team (Ally0 is player itself)
                else {
                    info[playerCounter + "_distanceToAlly_".concat(i + 1)] = distance;
                }

                i++;
            }

            i = 0;

            //Distance to other allies (index beings at 1, 0 is the current player)
            for (const otherPlayer of ct.members) {

                if (otherPlayer == null) {
                    i++;
                    continue;
                }

                //If this is the same player
                if (otherPlayer.userId == player.userId) {
                    i++;
                    continue;
                };

                let distance = this.calcVectorDistance(player.position, otherPlayer.position);

                //If in enemy team
                if (player.teamNumber != otherPlayer.teamNumber) {
                    info[playerCounter + "_distanceToEnemy_".concat(i)] = distance;
                }
                //If on allied team (Ally0 is player itself)
                else {
                    info[playerCounter + "_distanceToAlly_".concat(i + 1)] = distance;
                }

                i++;
            }

            playerCounter++;
        }

        this.playerInfoBuffer = this.playerInfoBuffer.concat(info);
    }

    /**
     * Calculates distance between two values
     *
     * @param {*} vecA
     * @param {*} vecB
     * @param {boolean} useTaxicab If true, the cheaper taxicab algorithm is used.
     * @returns
     * @memberof DemoFileParser
     */
    calcVectorDistance(vecA, vecB, useTaxicab = true) {
        if (useTaxicab) {
            let dX = Math.abs(vecA.x - vecB.x);
            let dY = Math.abs(vecA.y - vecB.y);
            let dZ = Math.abs(vecA.z - vecB.z);

            return dX + dY + dZ;
        } else {}
    }

    /**
     * Very start of the round, including buy time
     *
     * @memberof DemoFileParser
     */
    on_round_start() {
        this.ignoreTicks = true;
    }

    /**
     * Once the timer of the round starts and players can move
     * 
     */
    on_round_freeze_end() {
        //Start parsing ticks once player can move again
        this.ignoreTicks = false;
    }

    /**
     * The very end of the round, including time after timer ends
     *
     * @param {*} e
     * @memberof DemoFileParser
     */
    on_round_officially_ended() {
        const teams = this.demoFile.teams;

        const terrorists = teams[2];
        const cts = teams[3];

        const finishedRound = this.demoFile.gameRules.roundsPlayed;

        console.log(`ROUND >>${finishedRound}<< HAS ENDED!`);

        /** 
         * Write round intro all stored entries
         */
        this.deathDataBuffer.forEach((kill) => {
            kill.round = finishedRound;
        });
        this.playerInfoBuffer.forEach((playerInfo) => {
            playerInfo.round = finishedRound;
        });

        /** 
         * WRITE DATA TO CSV FILE
         */
        writerToPlayerInfo
            .writeRecords(this.playerInfoBuffer);
        writerToDeaths
            .writeRecords(this.deathDataBuffer);
        console.log(">> Data of last Round written to CSV");
=======
=======
>>>>>>> parent of 8abee5b... Update
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
<<<<<<< HEAD
>>>>>>> parent of 8abee5b... Update


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

=======


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

>>>>>>> parent of 8abee5b... Update
const vectorToString = (vector) => {
	const x = vector.x;
	const y = vector.y;
	const z = vector.z;

	return `"${x},${y},${z}"`
};