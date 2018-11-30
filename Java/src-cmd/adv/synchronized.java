public class ProducerAndConsumer {
	public static void main(String[] args) {
		Tickets t = new Tickets(6);
		new Consumer(t).start();
		new Producer(t).start()
	}
}

class Producer extends Thread {
	Tickets t = null;
	public Producer(Tickets t) { this.t = t;}

	public void run() {
		while (t.make_idx < t.size) t.put();

		System.out.println("Producer ends!")
	}
}

class Consumer extends Thread {
	Tickets t=null;
	public Consumer(Tickets t) { this.t = t;}

	public void run() {
		while (t.sell_idx < t.size) t.sell();

		System.out.println("Consumer buying ended");
	}
}

class Tickets {
	int size;
	int make_idx=0;
	int sell_idx=0;
	boolean available=false;
	public Tickets(int size) {this.size = size;}

	public synchronized void put() {
		System.out.println("Producer makes a ticket " + (++make_idx));
		available = true;
	}

	public synchronized void sell() {
		if (available==true && sell_idx<=make_idx)
			System.out.println("Consumer buys a ticket " + (++sell_idx));
		
		if(sell_idx==make_idx) available = false;
	}
}

